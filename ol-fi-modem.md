# Ol-Fi Modem Component Specification

## Objective
Create a simple, standalone Ol-Fi modem component that handles micrometer-scale chemical communication. The modem is a "dumb" device that only manages MVOC transmission/reception and basic protocol handling. All intelligence, logging, and analytics are handled by external systems (VMs) that connect to the modem.

## Core Modem Design

### 1. Simple Modem Class
```python
class OlFiModem:
    def __init__(self, cell_id, position=(0, 0)):
        # Hardware specs (fixed at initialization)
        self.cell_id = cell_id
        self.position = position
        self.transmission_power = 1.0     # MVOC release rate (ppm/tick)
        self.reception_sensitivity = 0.1  # Detection threshold (ppm)
        self.max_channels = 20           # Simultaneous MVOC types
        
        # Simple state
        self.outbound_queue = []         # Messages to send
        self.inbound_queue = []          # Received messages
        self.current_mvoc_levels = {}    # Current MVOC concentrations detected
        self.transmission_buffer = {}     # MVOCs being transmitted this tick
        
        # Basic configuration
        self.address = f"CELL_{cell_id:04d}"
        self.enabled = True
```

### 2. Core Modem Functions
```python
class OlFiModem:
    def transmit_message(self, target_address, message_type, payload):
        """Queue a message for transmission"""
        frame = self._encode_frame(target_address, message_type, payload)
        self.outbound_queue.append(frame)
    
    def receive_messages(self):
        """Return all received messages and clear buffer"""
        messages = self.inbound_queue.copy()
        self.inbound_queue.clear()
        return messages
    
    def update(self, chemical_environment):
        """Process one simulation tick"""
        # Receive: Sample chemical environment
        self._sample_environment(chemical_environment)
        self._decode_incoming_signals()
        
        # Transmit: Release queued MVOCs
        self._transmit_queued_messages(chemical_environment)
    
    def get_transmission_data(self):
        """Return current MVOC emissions for environment simulation"""
        return {
            'position': self.position,
            'mvocs': self.transmission_buffer.copy(),
            'power': self.transmission_power
        }
```

### 3. Message Protocol (Simplified)
```python
class ChemicalFrame:
    def __init__(self, target, msg_type, payload):
        self.sync_mvoc = "ACETOIN"           # Synchronization signal
        self.target_address = target         # 2-compound address encoding
        self.message_type = msg_type         # REQUEST/RESPONSE/BROADCAST
        self.payload = payload               # Actual data
        self.checksum = self._calculate_crc(payload)
        
    def to_mvoc_pattern(self):
        """Convert frame to MVOC concentration pattern"""
        mvoc_pattern = {}
        mvoc_pattern["ACETOIN"] = 1.0        # Sync signal
        mvoc_pattern.update(self._encode_address(self.target_address))
        mvoc_pattern.update(self._encode_type(self.message_type))
        mvoc_pattern.update(self._encode_payload(self.payload))
        mvoc_pattern.update(self._encode_checksum(self.checksum))
        return mvoc_pattern
```

### 4. MVOC Encoding/Decoding
```python
class OlFiModem:
    # Standard MVOC library for micro-scale communication
    MVOC_LIBRARY = {
        'sync': ['ACETOIN'],
        'address': ['ETHANOL', '2_METHYLBUTANOL', 'ISOAMYL_ACETATE', 'PHENYLETHANOL'],
        'control': ['ACETALDEHYDE', 'DIACETYL'],
        'data': ['HEXANOL', 'OCTANOL', 'DECANOL', 'DODECANOL', 
                'BUTANOIC_ACID', 'HEXANOIC_ACID', 'OCTANOIC_ACID'],
        'checksum': ['DIMETHYL_SULFIDE'],
        'terminator': ['BENZALDEHYDE']
    }
    
    def _encode_address(self, address):
        """Convert address to 2-compound MVOC pattern"""
        # Simple binary encoding using 4 address MVOCs
        addr_hash = hash(address) % 16  # 4-bit address
        pattern = {}
        for i, mvoc in enumerate(self.MVOC_LIBRARY['address']):
            pattern[mvoc] = 1.0 if (addr_hash >> i) & 1 else 0.0
        return pattern
    
    def _decode_address(self, mvoc_concentrations):
        """Extract target address from MVOC pattern"""
        addr_bits = 0
        for i, mvoc in enumerate(self.MVOC_LIBRARY['address']):
            if mvoc_concentrations.get(mvoc, 0) > self.reception_sensitivity:
                addr_bits |= (1 << i)
        return f"CELL_{addr_bits:04d}"
```

### 5. Interface for External VM Connection
```python
class ModemInterface:
    def __init__(self, modem):
        self.modem = modem
        self.external_handlers = {}  # Callbacks for different message types
        
    def register_handler(self, message_type, callback_function):
        """Register external VM handler for message type"""
        self.external_handlers[message_type] = callback_function
    
    def send_to_vm(self, message_data):
        """Send received message to connected VM"""
        message_type = message_data.get('type')
        if message_type in self.external_handlers:
            self.external_handlers[message_type](message_data)
    
    def receive_from_vm(self, target, msg_type, payload):
        """Receive command from VM to transmit"""
        self.modem.transmit_message(target, msg_type, payload)
    
    def get_status(self):
        """Return modem status for VM monitoring"""
        return {
            'cell_id': self.modem.cell_id,
            'position': self.modem.position,
            'queue_depth': len(self.modem.outbound_queue),
            'received_count': len(self.modem.inbound_queue),
            'current_mvocs': self.modem.current_mvoc_levels.copy(),
            'enabled': self.modem.enabled
        }
```

### 6. Chemical Environment Interface
```python
class ChemicalEnvironment:
    def __init__(self, dimensions=(100, 100)):
        self.width, self.height = dimensions
        self.mvoc_field = {}  # 3D array: [x][y][mvoc_type] = concentration
        self.diffusion_rate = 0.1
        self.decay_rate = 0.05
        
    def add_emission(self, position, mvoc_pattern, power):
        """Add MVOC emission from a modem"""
        x, y = position
        for mvoc_type, concentration in mvoc_pattern.items():
            if mvoc_type not in self.mvoc_field:
                self.mvoc_field[mvoc_type] = np.zeros((self.width, self.height))
            self.mvoc_field[mvoc_type][x, y] += concentration * power
    
    def sample_at_position(self, position):
        """Sample MVOC concentrations at a specific position"""
        x, y = position
        sample = {}
        for mvoc_type, field in self.mvoc_field.items():
            sample[mvoc_type] = field[x, y]
        return sample
    
    def update_diffusion(self):
        """Simple diffusion simulation (one tick)"""
        for mvoc_type in self.mvoc_field:
            # Apply diffusion and decay
            self.mvoc_field[mvoc_type] = self._diffuse_field(
                self.mvoc_field[mvoc_type]
            )
```

### 7. Simple Usage Example
```python
# Create modem
modem = OlFiModem(cell_id=123, position=(10, 15))
interface = ModemInterface(modem)

# Connect to external VM (simplified)
def handle_problem_request(message):
    # VM processes the computational request
    # VM sends response back through modem
    response_data = vm_process_request(message['payload'])
    interface.receive_from_vm(
        target=message['sender'],
        msg_type='RESPONSE',
        payload=response_data
    )

interface.register_handler('PROBLEM_REQUEST', handle_problem_request)

# Simulation loop
chemical_env = ChemicalEnvironment()

for tick in range(1000):
    # 1. Modem updates (all modems in population)
    modem.update(chemical_env)
    
    # 2. Send received messages to VM
    for message in modem.receive_messages():
        interface.send_to_vm(message)
    
    # 3. VM can send messages via interface
    if tick % 50 == 0:  # Every 50 ticks
        interface.receive_from_vm(
            target="CELL_0001",
            msg_type="STATUS_REQUEST", 
            payload={"query": "energy_level"}
        )
    
    # 4. Update chemical environment
    chemical_env.update_diffusion()
```

### 8. Standard Message Types
```python
MESSAGE_TYPES = {
    'PING': 'Connectivity test',
    'STATUS_REQUEST': 'Request cell status',
    'STATUS_RESPONSE': 'Cell status information',
    'PROBLEM_REQUEST': 'Request computational help',
    'PROBLEM_RESPONSE': 'Computational result',
    'RESOURCE_OFFER': 'Offer computational resources',
    'RESOURCE_ACCEPT': 'Accept resource offer',
    'EMERGENCY': 'Emergency shutdown/alert',
    'BROADCAST': 'General announcement'
}
```

## Key Design Principles

1. **Simplicity**: Modem only handles chemical I/O and basic protocol
2. **Stateless**: No complex state management or decision making
3. **Interface-Driven**: Clean API for external VM connection
4. **Protocol-Compliant**: Follows Ol-Fi standard for interoperability
5. **Environment-Agnostic**: Works with any chemical environment simulation
6. **Extensible**: Easy to add new MVOC types and message formats

The modem acts as a simple transceiver - it converts digital messages to/from chemical signals and provides a clean interface for external intelligent systems (VMs) to handle the actual computational logic, logging, and analytics.
