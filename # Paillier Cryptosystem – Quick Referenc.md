# Paillier Cryptosystem – Quick Reference
The **Paillier cryptosystem** is a public-key cryptosystem invented by Pascal Paillier in 1999. It's best known for supporting **additive homomorphic encryption**, which means you can perform addition on encrypted numbers without decrypting them first.

---

### 🔐 Key Features of Paillier Encryption

- **Additive Homomorphism**:  
  If you encrypt two numbers, \( E(m_1) \) and \( E(m_2) \), then:
  $$
  E(m_1) \cdot E(m_2) \mod n^2 = E(m_1 + m_2 \mod n)
  $$
  This allows secure computation of sums on encrypted data.

- **Probabilistic Encryption**:  
  Each encryption of the same plaintext yields a different ciphertext due to randomization, enhancing security.

- **Public-Key Based**:  
  Anyone can encrypt data using the public key, but only the holder of the private key can decrypt it.

---

### 🧮 How It Works (Simplified)

1. **Key Generation**:
   - Choose two large primes \( p \) and \( q \)
   - Compute \( n = pq \) and \( \lambda = \text{lcm}(p-1, q-1) \)
   - Choose \( g \in \mathbb{Z}_{n^2}^* \) such that \( \gcd(L(g^\lambda \mod n^2), n) = 1 \)
   - Public key: \( (n, g) \)
   - Private key: \( \lambda \)

2. **Encryption**:
   - Given message \( m \in \mathbb{Z}_n \), choose random \( r \in \mathbb{Z}_n^* \)
   - Compute ciphertext:  
     $$
     c = g^m \cdot r^n \mod n^2
     $$

3. **Decryption**:
   - Compute:  
     $$
     m = \frac{L(c^\lambda \mod n^2)}{L(g^\lambda \mod n^2)} \mod n
     $$
   - Where \( L(u) = \frac{u - 1}{n} \)
   - The decryption process works by:
     1. Raising the ciphertext \(c\) to the power of \(\lambda\) modulo \(n^2\)
     2. Applying the \(L\) function to get \(\frac{c^\lambda - 1}{n}\)
     3. Dividing by the precomputed value \(L(g^\lambda \mod n^2)\)
     4. Taking the result modulo \(n\) to recover the original message
   - This works because of the mathematical properties of the Carmichael function \(\lambda\) and the special structure of the Paillier cryptosystem

---

### 🧑‍💻 Use Cases

- **Secure voting systems**
- **Private data aggregation (e.g., surveys, statistics)**
- **Privacy-preserving machine learning**
- **Encrypted financial computations**

---

Would you like a working example in Java or Python to see it in action? Or are you interested in a specific application like secure voting or encrypted statistics?

> **Purpose**
> A public‑key scheme where **anyone** can encrypt with the public key `(n, g)`, but **only** the private key holder `λ` can decrypt. Supports additive homomorphism `E(m1)·E(m2) ≡ E(m1 + m2 mod n)`.

---

## 1. Background & intuition

Paillier leverages arithmetic in the multiplicative group of squares modulo `n²`. The trapdoor stems from knowing the *Carmichael* totient `λ = lcm(p‑1, q‑1)` of `n = pq`.

---

## 2. Notation

| Symbol              | Meaning                                                   |
| ------------------- | --------------------------------------------------------- |
| `p, q`              | large primes (same bit‑length)                            |
| `n = pq`            | RSA‑style modulus                                         |
| `λ = lcm(p−1, q−1)` | private Carmichael function                               |
| `g`                 | element of `ℤ*_{n²}` satisfying `gcd(L(g^λ mod n²), n)=1` |
| `L(u)`              | `(u − 1) / n` (maps `ℤ_{n²}` → `ℤ_n`)                     |
| `m ∈ ℤ_n`           | plaintext                                                 |
| `r ∈ ℤ*_{n}`        | fresh nonce per encryption                                |

---

## 3. Algorithms

### 3.1 Key Generation

1. Choose random  primes `p`, `q`; set `n = p q`.
2. Compute `λ = lcm(p−1, q−1)`.
3. Select `g ∈ ℤ*_{n²}` s.t. `gcd(L(g^λ mod n²), n)=1`.
4. **Public key** `pk = (n, g)`
   **Private key** `sk = λ`  *(optionally `μ = (L(g^λ mod n²))^{-1} mod n` pre‑computed)*.

### 3.2 Encryption

Given plaintext `m`:

1. Draw random `r ∈ ℤ*_n`.
2. Compute ciphertext
   $c = g^{m}\; r^{\;n} \pmod{n^{2}}$

### 3.3 Decryption

1. Compute $u = c^{\;λ} \bmod n^{2}$
2. Recover $m = \frac{L(u)}{L(g^{λ} \bmod n^{2})}\;\bmod n$
   (or `m = L(u) · μ mod n` if `μ` pre‑computed).

---

## 4. Correctness sketch

Euler–Carmichael guarantees `c^{λ} ≡ 1 + m n (mod n²)` when constructed as above; applying `L` removes the quadratic term, leaving `m` after scaling by the constant denominator.

---

## 5. Security considerations

* **Hard problem**: decisional composite residuosity (DCR).
* **Key sizes**: ≥ 2048‑bit `n` for 112‑bit security.
* **Randomness**: `r` **must** be uniform in `ℤ*_n`; reuse leaks `m`.
* **Encoding**: map arbitrary byte strings → integers < `n` (e.g., OAEP‑style padding).

---

## 6. Minimal Python reference
