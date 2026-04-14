import hashlib
import time

# =====================
# BLOCK CLASS
# =====================
class Block:
    def __init__(self, index, data, prev_hash):
        self.index = index
        self.timestamp = time.time()
        self.data = data
        self.prev_hash = prev_hash
        self.hash = self.create_hash()

    def create_hash(self):
        block_string = str(self.index) + str(self.timestamp) + str(self.data) + str(self.prev_hash)
        return hashlib.sha256(block_string.encode()).hexdigest()

# =====================
# BLOCKCHAIN CLASS
# =====================
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis = Block(0, "Genesis Block", "0")
        self.chain.append(genesis)

    def add_block(self, data):
        prev_block = self.chain[-1]
        new_block = Block(len(self.chain), data, prev_block.hash)
        self.chain.append(new_block)

    def show_chain(self):
        for block in self.chain:
            print("\n🔗 Block", block.index)
            print("Data:", block.data)
            print("Hash:", block.hash)

# =====================
# TEST RUN
# =====================
if __name__ == "__main__":
    bc = Blockchain()

    # example model data
    bc.add_block("Model Round 1 Weights")
    bc.add_block("Model Round 2 Weights")

    bc.show_chain()