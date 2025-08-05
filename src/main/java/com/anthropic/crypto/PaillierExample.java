
package com.anthropic.crypto;

import java.math.BigInteger;
import java.security.SecureRandom;

/**
 * Paillier Cryptosystem Implementation Example
 * 
 * This class demonstrates the Paillier cryptosystem, a probabilistic asymmetric algorithm
 * for public key cryptography that supports homomorphic addition operations.
 * 
 * The Paillier cryptosystem allows for:
 * - Encryption and decryption of messages
 * - Homomorphic addition of encrypted values
 * - Semantic security under chosen-plaintext attacks
 * 
 * @author Claude Agent Framework
 * @version 1.0
 */
public class PaillierExample {
    private static class PaillierKeyPair {
        public final BigInteger n;      // Public key component
        public final BigInteger g;      // Public key component
        public final BigInteger lambda; // Private key component
        
        public PaillierKeyPair(BigInteger n, BigInteger g, BigInteger lambda) {
            this.n = n;
            this.g = g;
            this.lambda = lambda;
        }
    }
    
    private static PaillierKeyPair generateKeyPair(int bitLength) {
        SecureRandom random = new SecureRandom();
        
        // Generate two large prime numbers
        BigInteger p = BigInteger.probablePrime(bitLength, random);
        BigInteger q = BigInteger.probablePrime(bitLength, random);
        
        // Calculate n = p * q
        BigInteger n = p.multiply(q);
        
        // Calculate lambda = lcm(p-1, q-1)
        BigInteger pMinus1 = p.subtract(BigInteger.ONE);
        BigInteger qMinus1 = q.subtract(BigInteger.ONE);
        BigInteger lambda = pMinus1.multiply(qMinus1)
                .divide(pMinus1.gcd(qMinus1));
        
        // Choose g = n + 1 (a common choice that works well)
        BigInteger g = n.add(BigInteger.ONE);
        
        return new PaillierKeyPair(n, g, lambda);
    }
    
    private static BigInteger encrypt(BigInteger message, BigInteger n, BigInteger g) {
        SecureRandom random = new SecureRandom();
        
        // Choose random r in Z_n*
        BigInteger r = new BigInteger(n.bitLength(), random);
        while (r.compareTo(n) >= 0 || r.gcd(n).compareTo(BigInteger.ONE) != 0) {
            r = new BigInteger(n.bitLength(), random);
        }
        
        // Compute c = g^m * r^n mod n^2
        BigInteger nSquared = n.multiply(n);
        BigInteger gPowM = g.modPow(message, nSquared);
        BigInteger rPowN = r.modPow(n, nSquared);
        
        return gPowM.multiply(rPowN).mod(nSquared);
    }
    
    private static BigInteger decrypt(BigInteger ciphertext, BigInteger n, BigInteger lambda) {
        BigInteger nSquared = n.multiply(n);
        
        // Compute L(c^lambda mod n^2)
        BigInteger cPowLambda = ciphertext.modPow(lambda, nSquared);
        BigInteger lC = cPowLambda.subtract(BigInteger.ONE).divide(n);
        
        // Compute L(g^lambda mod n^2)
        BigInteger gPowLambda = n.add(BigInteger.ONE).modPow(lambda, nSquared);
        BigInteger lG = gPowLambda.subtract(BigInteger.ONE).divide(n);
        
        // Compute m = L(c^lambda mod n^2) / L(g^lambda mod n^2) mod n
        return lC.multiply(lG.modInverse(n)).mod(n);
    }
    
    public static void main(String[] args) {
        // Generate key pair
        PaillierKeyPair keyPair = generateKeyPair(512);
        
        // Original message
        BigInteger message = new BigInteger("42");
        System.out.println("Original message: " + message);
        
        // Encrypt the message
        BigInteger ciphertext = encrypt(message, keyPair.n, keyPair.g);
        System.out.println("Encrypted message: " + ciphertext);
        
        // Decrypt the message
        BigInteger decrypted = decrypt(ciphertext, keyPair.n, keyPair.lambda);
        System.out.println("Decrypted message: " + decrypted);
        
        // Demonstrate homomorphic addition
        BigInteger message2 = new BigInteger("58");
        BigInteger ciphertext2 = encrypt(message2, keyPair.n, keyPair.g);
        
        // Add the ciphertexts
        BigInteger sumCiphertext = ciphertext.multiply(ciphertext2).mod(keyPair.n.multiply(keyPair.n));
        
        // Decrypt the sum
        BigInteger decryptedSum = decrypt(sumCiphertext, keyPair.n, keyPair.lambda);
        System.out.println("\nHomomorphic addition demonstration:");
        System.out.println("Message 1: " + message);
        System.out.println("Message 2: " + message2);
        System.out.println("Sum of messages: " + message.add(message2));
        System.out.println("Decrypted sum: " + decryptedSum);
    }
}
