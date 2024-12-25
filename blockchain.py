import time
import hashlib
from datetime import datetime
import qrcode
import base64
from io import BytesIO

class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_data = []
        self.difficulty = 4  # Number of leading zeros required in hash
        self.create_block(proof=1, previous_hash='0', mining_time=0)

    def create_block(self, proof, previous_hash, mining_time):
        """
        Create a block and add it to the chain.
        Now with separate difficulty and hash values.
        """
        guess = f'{proof}{previous_hash}'.encode()
        pow_hash = hashlib.sha256(guess).hexdigest()
        
        block = {
            'index': len(self.chain) + 1,
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'data': self.pending_data,
            'proof': proof,
            'previous_hash': previous_hash,
            'mining_time': f"{mining_time:.2f} seconds",
            'difficulty': self.difficulty,  # Explicit difficulty value
            'pow_hash': pow_hash  # Separate PoW hash
        }
        
        self.pending_data = []
        self.chain.append(block)
        return block

    def mine_block(self, previous_hash):
        """
        Mine a new block using Proof of Work.
        """
        mining_start = time.time()
        proof = 0
        while not self.is_valid_proof(proof, previous_hash):
            proof += 1
        mining_time = time.time() - mining_start
        return proof, mining_time

    def create_block(self, proof, previous_hash, mining_time):
        """
        Create a block and add it to the chain.
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'data': self.pending_data,
            'proof': proof,
            'previous_hash': previous_hash,
            'difficulty': self.difficulty,  # Add difficulty level
            'mining_time': f"{mining_time:.2f} seconds",
            'pow_hash': self.calculate_hash(proof, previous_hash)  # Add PoW hash
        }
        
        self.pending_data = []
        self.chain.append(block)
        return block

    def calculate_hash(self, proof, previous_hash):
        """Calculate PoW hash"""
        guess = f'{proof}{previous_hash}'.encode()
        return hashlib.sha256(guess).hexdigest()

    def is_valid_proof(self, proof, previous_hash):
        """
        Check if a proof is valid by verifying it produces a hash with the required
        number of leading zeros when combined with the previous hash
        """
        hash_value = self.calculate_hash(proof, previous_hash)
        return hash_value[:self.difficulty] == '0' * self.difficulty

    def add_data(self, data):
        """Add ESG data to the list of pending data."""
        self.pending_data.append(data)

    def get_previous_block(self):
        """Return the last block in the chain."""
        return self.chain[-1] if self.chain else None

    def hash(self, block):
        """Create a SHA-256 hash of a block."""
        # Convert block to string and encode
        block_string = str(block).encode()
        return hashlib.sha256(block_string).hexdigest()

    def is_chain_valid(self):
        """
        Check if the blockchain is valid.
        Verifies both the chain of hashes and the proof of work for each block.
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Check if the current block's previous_hash matches the hash of the previous block
            if current_block['previous_hash'] != self.hash(previous_block):
                return False

            # Verify the proof of work
            if not self.is_valid_proof(current_block['proof'], 
                                     current_block['previous_hash']):
                return False

        return True

    def add_new_block(self, data):
        """
        Convenience method to add data and create a new block in one step.
        Includes the mining process and accurate timing.
        """
        self.add_data(data)
        previous_block = self.get_previous_block()
        previous_hash = self.hash(previous_block) if previous_block else '0'
        
        # Mine the block and get both proof and mining time
        proof, mining_time = self.mine_block(previous_hash)
        
        # Create and return the new block with the mining time
        return self.create_block(proof, previous_hash, mining_time)

    def get_block_by_index(self, index):
        """
        Retrieve a block by its index.
        """
        for block in self.chain:
            if block['index'] == index:
                return block
        return None

    def verify_block(self, block_index):
        """
        Verify a specific block's integrity
        Returns tuple (is_valid, details)
        """
        block = self.get_block_by_index(block_index)
        if not block:
            return False, "Block not found"
            
        if block_index == 1:  # Genesis block
            return True, "Genesis block is valid"
            
        previous_block = self.get_block_by_index(block_index - 1)
        if not previous_block:
            return False, "Previous block not found"

        # Verify hash chain
        if block['previous_hash'] != self.hash(previous_block):
            return False, "Invalid hash chain"

        # Verify proof of work
        if not self.is_valid_proof(block['proof'], block['previous_hash']):
            return False, "Invalid proof of work"

        return True, "Block is valid"

    def generate_verification_qr(self, block):
        """Generate QR code for block verification"""
        try:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            
            # Create verification data URL
            verification_url = f"block-{block['index']}"
            qr.add_data(verification_url)
            qr.make(fit=True)

            # Create QR code image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"QR Code generation error: {str(e)}")
            return ""