package com.anthropic.crypto;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Timeout;
import static org.junit.jupiter.api.Assertions.*;

import java.math.BigInteger;
import java.lang.reflect.Method;
import java.lang.reflect.Constructor;
import java.util.concurrent.TimeUnit;

/**
 * Comprehensive test suite for PaillierExample - a leaf node cryptographic implementation.
 * Tests key generation, encryption/decryption, homomorphic properties, and security aspects.
 */
@DisplayName("PaillierExample Cryptographic Tests")
class PaillierExampleTest {

    private static final int TEST_KEY_SIZE = 512; // Smaller key size for faster tests
    private static final BigInteger TEST_MESSAGE_1 = new BigInteger("42");
    private static final BigInteger TEST_MESSAGE_2 = new BigInteger("58");
    private static final BigInteger ZERO = BigInteger.ZERO;
    private static final BigInteger ONE = BigInteger.ONE;

    // Using reflection to access private methods for thorough testing
    private Method generateKeyPairMethod;
    private Method encryptMethod;
    private Method decryptMethod;
    private Constructor<?> keyPairConstructor;

    @BeforeEach
    void setUp() throws Exception {
        // Get access to private methods for testing
        Class<?> paillierClass = PaillierExample.class;
        generateKeyPairMethod = paillierClass.getDeclaredMethod("generateKeyPair", int.class);
        generateKeyPairMethod.setAccessible(true);
        
        encryptMethod = paillierClass.getDeclaredMethod("encrypt", BigInteger.class, BigInteger.class, BigInteger.class);
        encryptMethod.setAccessible(true);
        
        decryptMethod = paillierClass.getDeclaredMethod("decrypt", BigInteger.class, BigInteger.class, BigInteger.class);
        decryptMethod.setAccessible(true);

        // Get access to PaillierKeyPair constructor
        Class<?>[] innerClasses = paillierClass.getDeclaredClasses();
        for (Class<?> innerClass : innerClasses) {
            if (innerClass.getSimpleName().equals("PaillierKeyPair")) {
                keyPairConstructor = innerClass.getDeclaredConstructor(BigInteger.class, BigInteger.class, BigInteger.class);
                keyPairConstructor.setAccessible(true);
                break;
            }
        }
    }

    @Nested
    @DisplayName("Key Generation Tests")
    class KeyGenerationTests {

        @Test
        @DisplayName("Should generate valid key pair with correct bit length")
        @Timeout(value = 10, unit = TimeUnit.SECONDS)
        void shouldGenerateValidKeyPair() throws Exception {
            Object keyPair = generateKeyPairMethod.invoke(null, TEST_KEY_SIZE);
            
            assertNotNull(keyPair);
            
            // Use reflection to access key components
            Class<?> keyPairClass = keyPair.getClass();
            BigInteger n = (BigInteger) keyPairClass.getField("n").get(keyPair);
            BigInteger g = (BigInteger) keyPairClass.getField("g").get(keyPair);
            BigInteger lambda = (BigInteger) keyPairClass.getField("lambda").get(keyPair);
            
            assertNotNull(n, "n should not be null");
            assertNotNull(g, "g should not be null");
            assertNotNull(lambda, "lambda should not be null");
            
            // Verify n has approximately the right bit length (should be ~2 * TEST_KEY_SIZE)
            assertTrue(n.bitLength() >= TEST_KEY_SIZE * 2 - 10, "n should have approximately 2 * key_size bits");
            assertTrue(n.bitLength() <= TEST_KEY_SIZE * 2 + 10, "n should not exceed expected bit length");
            
            // Verify g = n + 1 (common choice)
            assertEquals(n.add(ONE), g, "g should equal n + 1");
            
            // Verify lambda is positive and less than n
            assertTrue(lambda.compareTo(ZERO) > 0, "lambda should be positive");
            assertTrue(lambda.compareTo(n) < 0, "lambda should be less than n");
        }

        @RepeatedTest(5)
        @DisplayName("Should generate different keys on repeated calls")
        void shouldGenerateDifferentKeys() throws Exception {
            Object keyPair1 = generateKeyPairMethod.invoke(null, TEST_KEY_SIZE);
            Object keyPair2 = generateKeyPairMethod.invoke(null, TEST_KEY_SIZE);
            
            Class<?> keyPairClass = keyPair1.getClass();
            BigInteger n1 = (BigInteger) keyPairClass.getField("n").get(keyPair1);
            BigInteger n2 = (BigInteger) keyPairClass.getField("n").get(keyPair2);
            
            assertNotEquals(n1, n2, "Different key generation calls should produce different keys");
        }

        @Test
        @DisplayName("Should handle different key sizes")
        void shouldHandleDifferentKeySizes() throws Exception {
            int[] keySizes = {256, 512, 1024};
            
            for (int keySize : keySizes) {
                Object keyPair = generateKeyPairMethod.invoke(null, keySize);
                assertNotNull(keyPair, "Should generate key pair for size " + keySize);
                
                Class<?> keyPairClass = keyPair.getClass();
                BigInteger n = (BigInteger) keyPairClass.getField("n").get(keyPair);
                
                // Allow some tolerance in bit length due to prime generation
                assertTrue(n.bitLength() >= keySize * 2 - 20, 
                    "Key size " + keySize + " should produce n with appropriate bit length");
            }
        }
    }

    @Nested
    @DisplayName("Encryption and Decryption Tests")
    class EncryptionDecryptionTests {

        private Object keyPair;
        private BigInteger n, g, lambda;

        @BeforeEach
        void setUpKeys() throws Exception {
            keyPair = generateKeyPairMethod.invoke(null, TEST_KEY_SIZE);
            Class<?> keyPairClass = keyPair.getClass();
            n = (BigInteger) keyPairClass.getField("n").get(keyPair);
            g = (BigInteger) keyPairClass.getField("g").get(keyPair);
            lambda = (BigInteger) keyPairClass.getField("lambda").get(keyPair);
        }

        @Test
        @DisplayName("Should encrypt and decrypt basic messages correctly")
        void shouldEncryptAndDecryptBasicMessages() throws Exception {
            BigInteger ciphertext = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_1, n, g);
            assertNotNull(ciphertext, "Encryption should produce a ciphertext");
            
            BigInteger decrypted = (BigInteger) decryptMethod.invoke(null, ciphertext, n, lambda);
            assertEquals(TEST_MESSAGE_1, decrypted, "Decryption should recover original message");
        }

        @Test
        @DisplayName("Should handle zero message")
        void shouldHandleZeroMessage() throws Exception {
            BigInteger ciphertext = (BigInteger) encryptMethod.invoke(null, ZERO, n, g);
            BigInteger decrypted = (BigInteger) decryptMethod.invoke(null, ciphertext, n, lambda);
            assertEquals(ZERO, decrypted, "Should correctly encrypt and decrypt zero");
        }

        @Test
        @DisplayName("Should handle one message")
        void shouldHandleOneMessage() throws Exception {
            BigInteger ciphertext = (BigInteger) encryptMethod.invoke(null, ONE, n, g);
            BigInteger decrypted = (BigInteger) decryptMethod.invoke(null, ciphertext, n, lambda);
            assertEquals(ONE, decrypted, "Should correctly encrypt and decrypt one");
        }

        @Test
        @DisplayName("Should handle large messages within modulus")
        void shouldHandleLargeMessages() throws Exception {
            BigInteger largeMessage = n.subtract(ONE); // Largest valid message
            BigInteger ciphertext = (BigInteger) encryptMethod.invoke(null, largeMessage, n, g);
            BigInteger decrypted = (BigInteger) decryptMethod.invoke(null, ciphertext, n, lambda);
            assertEquals(largeMessage, decrypted, "Should correctly handle large messages");
        }

        @RepeatedTest(10)
        @DisplayName("Should produce different ciphertexts for same message (probabilistic)")
        void shouldProduceDifferentCiphertexts() throws Exception {
            BigInteger ciphertext1 = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_1, n, g);
            BigInteger ciphertext2 = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_1, n, g);
            
            assertNotEquals(ciphertext1, ciphertext2, 
                "Paillier encryption should be probabilistic - same message should produce different ciphertexts");
            
            // But both should decrypt to the same message
            BigInteger decrypted1 = (BigInteger) decryptMethod.invoke(null, ciphertext1, n, lambda);
            BigInteger decrypted2 = (BigInteger) decryptMethod.invoke(null, ciphertext2, n, lambda);
            
            assertEquals(TEST_MESSAGE_1, decrypted1, "First ciphertext should decrypt correctly");
            assertEquals(TEST_MESSAGE_1, decrypted2, "Second ciphertext should decrypt correctly");
        }

        @Test
        @DisplayName("Should produce ciphertexts in valid range")
        void shouldProduceCiphertextsInValidRange() throws Exception {
            BigInteger ciphertext = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_1, n, g);
            BigInteger nSquared = n.multiply(n);
            
            assertTrue(ciphertext.compareTo(ZERO) > 0, "Ciphertext should be positive");
            assertTrue(ciphertext.compareTo(nSquared) < 0, "Ciphertext should be less than n^2");
        }
    }

    @Nested
    @DisplayName("Homomorphic Addition Tests")
    class HomomorphicAdditionTests {

        private Object keyPair;
        private BigInteger n, g, lambda;

        @BeforeEach
        void setUpKeys() throws Exception {
            keyPair = generateKeyPairMethod.invoke(null, TEST_KEY_SIZE);
            Class<?> keyPairClass = keyPair.getClass();
            n = (BigInteger) keyPairClass.getField("n").get(keyPair);
            g = (BigInteger) keyPairClass.getField("g").get(keyPair);
            lambda = (BigInteger) keyPairClass.getField("lambda").get(keyPair);
        }

        @Test
        @DisplayName("Should correctly perform homomorphic addition")
        void shouldPerformHomomorphicAddition() throws Exception {
            // Encrypt both messages
            BigInteger ciphertext1 = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_1, n, g);
            BigInteger ciphertext2 = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_2, n, g);
            
            // Perform homomorphic addition: multiply ciphertexts
            BigInteger nSquared = n.multiply(n);
            BigInteger sumCiphertext = ciphertext1.multiply(ciphertext2).mod(nSquared);
            
            // Decrypt the sum
            BigInteger decryptedSum = (BigInteger) decryptMethod.invoke(null, sumCiphertext, n, lambda);
            BigInteger expectedSum = TEST_MESSAGE_1.add(TEST_MESSAGE_2);
            
            assertEquals(expectedSum, decryptedSum, 
                "Homomorphic addition should produce the sum of the original messages");
        }

        @Test
        @DisplayName("Should handle homomorphic addition with zero")
        void shouldHandleHomomorphicAdditionWithZero() throws Exception {
            BigInteger ciphertext1 = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_1, n, g);
            BigInteger ciphertextZero = (BigInteger) encryptMethod.invoke(null, ZERO, n, g);
            
            BigInteger nSquared = n.multiply(n);
            BigInteger sumCiphertext = ciphertext1.multiply(ciphertextZero).mod(nSquared);
            
            BigInteger decryptedSum = (BigInteger) decryptMethod.invoke(null, sumCiphertext, n, lambda);
            
            assertEquals(TEST_MESSAGE_1, decryptedSum, 
                "Adding zero homomorphically should not change the original message");
        }

        @Test
        @DisplayName("Should handle multiple homomorphic additions")
        void shouldHandleMultipleHomomorphicAdditions() throws Exception {
            BigInteger[] messages = {
                new BigInteger("10"),
                new BigInteger("20"),
                new BigInteger("30"),
                new BigInteger("40")
            };
            
            BigInteger expectedSum = ZERO;
            BigInteger ciphertextProduct = ONE;
            BigInteger nSquared = n.multiply(n);
            
            // Encrypt each message and multiply ciphertexts
            for (BigInteger message : messages) {
                expectedSum = expectedSum.add(message);
                BigInteger ciphertext = (BigInteger) encryptMethod.invoke(null, message, n, g);
                ciphertextProduct = ciphertextProduct.multiply(ciphertext).mod(nSquared);
            }
            
            // Decrypt the product
            BigInteger decryptedSum = (BigInteger) decryptMethod.invoke(null, ciphertextProduct, n, lambda);
            
            assertEquals(expectedSum, decryptedSum, 
                "Multiple homomorphic additions should produce the correct sum");
        }

        @Test
        @DisplayName("Should be commutative")
        void shouldBeCommutative() throws Exception {
            BigInteger ciphertext1 = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_1, n, g);
            BigInteger ciphertext2 = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_2, n, g);
            
            BigInteger nSquared = n.multiply(n);
            BigInteger sum1 = ciphertext1.multiply(ciphertext2).mod(nSquared);
            BigInteger sum2 = ciphertext2.multiply(ciphertext1).mod(nSquared);
            
            BigInteger decrypted1 = (BigInteger) decryptMethod.invoke(null, sum1, n, lambda);
            BigInteger decrypted2 = (BigInteger) decryptMethod.invoke(null, sum2, n, lambda);
            
            assertEquals(decrypted1, decrypted2, "Homomorphic addition should be commutative");
        }

        @Test
        @DisplayName("Should be associative")
        void shouldBeAssociative() throws Exception {
            BigInteger message3 = new BigInteger("15");
            
            BigInteger ciphertext1 = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_1, n, g);
            BigInteger ciphertext2 = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_2, n, g);
            BigInteger ciphertext3 = (BigInteger) encryptMethod.invoke(null, message3, n, g);
            
            BigInteger nSquared = n.multiply(n);
            
            // (c1 * c2) * c3
            BigInteger temp1 = ciphertext1.multiply(ciphertext2).mod(nSquared);
            BigInteger result1 = temp1.multiply(ciphertext3).mod(nSquared);
            
            // c1 * (c2 * c3)
            BigInteger temp2 = ciphertext2.multiply(ciphertext3).mod(nSquared);
            BigInteger result2 = ciphertext1.multiply(temp2).mod(nSquared);
            
            BigInteger decrypted1 = (BigInteger) decryptMethod.invoke(null, result1, n, lambda);
            BigInteger decrypted2 = (BigInteger) decryptMethod.invoke(null, result2, n, lambda);
            
            assertEquals(decrypted1, decrypted2, "Homomorphic addition should be associative");
        }
    }

    @Nested
    @DisplayName("Security and Edge Case Tests")
    class SecurityAndEdgeCaseTests {

        private Object keyPair;
        private BigInteger n, g, lambda;

        @BeforeEach
        void setUpKeys() throws Exception {
            keyPair = generateKeyPairMethod.invoke(null, TEST_KEY_SIZE);
            Class<?> keyPairClass = keyPair.getClass();
            n = (BigInteger) keyPairClass.getField("n").get(keyPair);
            g = (BigInteger) keyPairClass.getField("g").get(keyPair);
            lambda = (BigInteger) keyPairClass.getField("lambda").get(keyPair);
        }

        @Test
        @DisplayName("Should handle messages at modulus boundary")
        void shouldHandleMessagesAtModulusBoundary() throws Exception {
            BigInteger boundaryMessage = n.subtract(ONE);
            
            BigInteger ciphertext = (BigInteger) encryptMethod.invoke(null, boundaryMessage, n, g);
            BigInteger decrypted = (BigInteger) decryptMethod.invoke(null, ciphertext, n, lambda);
            
            assertEquals(boundaryMessage, decrypted, 
                "Should correctly handle messages at the modulus boundary");
        }

        @Test
        @DisplayName("Should handle message reduction modulo n")
        void shouldHandleMessageReductionModuloN() throws Exception {
            BigInteger largeMessage = n.add(TEST_MESSAGE_1); // Message > n
            BigInteger expectedReduced = TEST_MESSAGE_1; // Should be reduced to TEST_MESSAGE_1
            
            BigInteger ciphertext = (BigInteger) encryptMethod.invoke(null, largeMessage, n, g);
            BigInteger decrypted = (BigInteger) decryptMethod.invoke(null, ciphertext, n, lambda);
            
            assertEquals(expectedReduced, decrypted, 
                "Messages should be automatically reduced modulo n");
        }

        @Test
        @DisplayName("Should maintain semantic security properties")
        void shouldMaintainSemanticSecurityProperties() throws Exception {
            // Encrypt the same message multiple times and verify all ciphertexts are different
            BigInteger[] ciphertexts = new BigInteger[10];
            
            for (int i = 0; i < ciphertexts.length; i++) {
                ciphertexts[i] = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_1, n, g);
            }
            
            // Verify all ciphertexts are different (probabilistic encryption)
            for (int i = 0; i < ciphertexts.length; i++) {
                for (int j = i + 1; j < ciphertexts.length; j++) {
                    assertNotEquals(ciphertexts[i], ciphertexts[j], 
                        "All ciphertexts should be different due to probabilistic encryption");
                }
            }
            
            // But all should decrypt to the same message
            for (BigInteger ciphertext : ciphertexts) {
                BigInteger decrypted = (BigInteger) decryptMethod.invoke(null, ciphertext, n, lambda);
                assertEquals(TEST_MESSAGE_1, decrypted, 
                    "All ciphertexts should decrypt to the original message");
            }
        }

        @Test
        @DisplayName("Should have ciphertext expansion property")
        void shouldHaveCiphertextExpansionProperty() throws Exception {
            BigInteger ciphertext = (BigInteger) encryptMethod.invoke(null, TEST_MESSAGE_1, n, g);
            
            // Ciphertext should be significantly larger than the original message
            assertTrue(ciphertext.bitLength() > TEST_MESSAGE_1.bitLength() * 2, 
                "Ciphertext should be much larger than original message (ciphertext expansion)");
            
            // Ciphertext should be in the range [1, n^2)
            BigInteger nSquared = n.multiply(n);
            assertTrue(ciphertext.compareTo(ONE) >= 0, "Ciphertext should be at least 1");
            assertTrue(ciphertext.compareTo(nSquared) < 0, "Ciphertext should be less than n^2");
        }
    }

    @Nested
    @DisplayName("Integration Tests")
    class IntegrationTests {

        @Test
        @DisplayName("Should demonstrate complete Paillier workflow")
        @Timeout(value = 30, unit = TimeUnit.SECONDS)
        void shouldDemonstrateCompleteWorkflow() throws Exception {
            // This test simulates the main method workflow
            Object keyPair = generateKeyPairMethod.invoke(null, 512);
            Class<?> keyPairClass = keyPair.getClass();
            BigInteger n = (BigInteger) keyPairClass.getField("n").get(keyPair);
            BigInteger g = (BigInteger) keyPairClass.getField("g").get(keyPair);
            BigInteger lambda = (BigInteger) keyPairClass.getField("lambda").get(keyPair);
            
            // Test basic encryption/decryption
            BigInteger message = new BigInteger("42");
            BigInteger ciphertext = (BigInteger) encryptMethod.invoke(null, message, n, g);
            BigInteger decrypted = (BigInteger) decryptMethod.invoke(null, ciphertext, n, lambda);
            assertEquals(message, decrypted, "Basic encryption/decryption should work");
            
            // Test homomorphic addition
            BigInteger message2 = new BigInteger("58");
            BigInteger ciphertext2 = (BigInteger) encryptMethod.invoke(null, message2, n, g);
            
            BigInteger nSquared = n.multiply(n);
            BigInteger sumCiphertext = ciphertext.multiply(ciphertext2).mod(nSquared);
            BigInteger decryptedSum = (BigInteger) decryptMethod.invoke(null, sumCiphertext, n, lambda);
            
            assertEquals(message.add(message2), decryptedSum, "Homomorphic addition should work");
            
            // Verify the workflow produces expected results
            assertEquals(new BigInteger("100"), decryptedSum, "42 + 58 should equal 100");
        }

        @Test
        @DisplayName("Should handle real-world use case scenario")
        void shouldHandleRealWorldUseCase() throws Exception {
            // Simulate a voting scenario where votes are encrypted and tallied homomorphically
            Object keyPair = generateKeyPairMethod.invoke(null, 512);
            Class<?> keyPairClass = keyPair.getClass();
            BigInteger n = (BigInteger) keyPairClass.getField("n").get(keyPair);
            BigInteger g = (BigInteger) keyPairClass.getField("g").get(keyPair);
            BigInteger lambda = (BigInteger) keyPairClass.getField("lambda").get(keyPair);
            
            // Simulate 10 votes (0 or 1)
            int[] votes = {1, 0, 1, 1, 0, 1, 0, 1, 1, 0}; // 6 yes, 4 no
            int expectedYesVotes = 6;
            
            BigInteger tallyProduct = ONE;
            BigInteger nSquared = n.multiply(n);
            
            for (int vote : votes) {
                BigInteger encryptedVote = (BigInteger) encryptMethod.invoke(null, BigInteger.valueOf(vote), n, g);
                tallyProduct = tallyProduct.multiply(encryptedVote).mod(nSquared);
            }
            
            BigInteger finalTally = (BigInteger) decryptMethod.invoke(null, tallyProduct, n, lambda);
            
            assertEquals(BigInteger.valueOf(expectedYesVotes), finalTally, 
                "Homomorphic tallying should correctly count encrypted votes");
        }
    }
}