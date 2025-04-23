import numpy as np 

class RandomCliffordCircuit():
    def __init__(self, n_logical_qubits, depth, circuit_index):
        self.depth = depth 
        self.n_logical_qubits = n_logical_qubits
        self.circuit_index = circuit_index
        if circuit_index in ['3', '4']:
            self.single_qubit_gate_list = ['I', 'X', 'Y', 'Z', 'H']
        else:
            raise ValueError('Invalid circuit index')
        self.include_two_qubit_gates = circuit_index in ['4']
        self.sq_gates = None
        self.tq_gates = None

    def sample_circuit(self):
        self.sq_gates = np.random.choice(self.single_qubit_gate_list, size=(self.depth, self.n_logical_qubits))
        if self.include_two_qubit_gates:
            self.tq_gates = np.stack([np.arange(self.n_logical_qubits)]*self.depth, axis=0)
            for d in range(self.depth):
                np.random.shuffle(self.tq_gates[d])

    def load_circuit(self, sq_gates, tq_gates=None):
        assert sq_gates.shape == (self.depth, self.n_logical_qubits)
        if not self.include_two_qubit_gates: assert tq_gates is None  
        self.sq_gates = sq_gates
        self.tq_gates = tq_gates

    def to_string_lines(self):
        raise NotImplementedError

    @classmethod
    def from_lines(cls, lines):
        raise NotImplementedError


def append_circuit_to_file(circuit, file_path):
    with open(file_path, 'a') as f:
        f.write(f"===CIRCUIT T{circuit.circuit_index}===\n")
        f.write('\n'.join(circuit.to_string_lines()) + '\n')


def load_circuit_from_file(file_path, index):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()

    circuit_starts = [i for i, line in enumerate(lines) if line.startswith("===CIRCUIT")]
    if index >= len(circuit_starts):
        raise IndexError("Circuit index out of range")

    start = circuit_starts[index]
    end = circuit_starts[index + 1] if index + 1 < len(circuit_starts) else len(lines)
    header = lines[start]
    body = lines[start + 1:end]

    if header == "===CIRCUIT T3===":
        return TypeICircuit.from_lines(body)
    elif header == "===CIRCUIT T4===":
        return TypeIICircuit.from_lines(body)
    else:
        raise ValueError("Unknown circuit type")


class TypeICircuit(RandomCliffordCircuit):
    def __iter__(self):
        self._schedule = []
        for t in range(self.depth):
            for q in range(self.n_logical_qubits):
                self._schedule.append((t, q))
        for t in reversed(range(self.depth)):
            for q in range(self.n_logical_qubits):
                self._schedule.append((t, q))
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self._schedule):
            raise StopIteration
        t, q = self._schedule[self._index]
        gate = self.sq_gates[t, q]
        self._index += 1
        return gate, t, [q]

    def to_string_lines(self):
        return [''.join(self.sq_gates[t]) for t in range(self.depth)]

    @classmethod
    def from_lines(cls, lines):
        depth = len(lines)
        n_logical_qubits = len(lines[0])
        sq_gates = np.array([list(line.strip()) for line in lines], dtype='<U2')
        circuit = cls(n_logical_qubits, depth, circuit_index='3')
        circuit.load_circuit(sq_gates)
        return circuit


class TypeIICircuit(RandomCliffordCircuit):
    def __iter__(self):
        self._layer_seq = []
        for t in range(self.depth):
            self._layer_seq.append(('sq', t))
            self._layer_seq.append(('tq', t))
        for t in reversed(range(self.depth)):
            self._layer_seq.append(('tq', t))
            self._layer_seq.append(('sq', t))
        self._layer_idx = 0
        self._q = 0
        return self

    def __next__(self):
        while self._layer_idx < len(self._layer_seq):
            layer_type, logical_t = self._layer_seq[self._layer_idx]
            layer_time = self._layer_idx

            if layer_type == 'sq':
                if self._q < self.n_logical_qubits:
                    gate = self.sq_gates[logical_t, self._q]
                    result = (gate, layer_time, [self._q])
                    self._q += 1
                    return result
                else:
                    self._q = 0
                    self._layer_idx += 1

            elif layer_type == 'tq':
                if self._q < self.n_logical_qubits:
                    targets = [
                        self.tq_gates[logical_t][self._q],
                        self.tq_gates[logical_t][self._q + 1]
                    ]
                    result = ('CX', layer_time, targets)
                    self._q += 2
                    return result
                else:
                    self._q = 0
                    self._layer_idx += 1

        raise StopIteration

    def to_string_lines(self):
        lines = []
        for t in range(self.depth):
            lines.append(''.join(self.sq_gates[t]))
            lines.append(''.join(str(i) for i in self.tq_gates[t]))
        return lines

    @classmethod
    def from_lines(cls, lines):
        assert len(lines) % 2 == 0, "Expected even number of lines for SQ+TQ pairs"
        depth = len(lines) // 2
        n_logical_qubits = len(lines[0])
        sq_gates = np.empty((depth, n_logical_qubits), dtype='<U2')
        tq_gates = np.empty((depth, n_logical_qubits), dtype=int)
        for t in range(depth):
            sq_gates[t] = list(lines[2 * t].strip())
            tq_gates[t] = [int(c) for c in lines[2 * t + 1].strip()]
        circuit = cls(n_logical_qubits, depth, circuit_index='4')
        circuit.load_circuit(sq_gates, tq_gates)
        return circuit
