#!/usr/bin/env runhaskell

-- Probabilistic Quantum Circuits Example
-- Demonstrates quantum measurement with uncertainty modeling

-- | Probabilistic quantum measurement
probabilisticMeasurement :: IO ()
probabilisticMeasurement = do
    putStrLn "=== Probabilistic Quantum Measurement ==="
    
    -- Create a quantum state
    let q = qubit 0.4
    putStrLn $ "Initial quantum state: " ++ show q
    
    -- Apply Hadamard gate
    let hq = hadamard q
    putStrLn $ "After Hadamard gate: " ++ show hq
    
    -- Multiple measurements to show probabilistic nature
    measurements <- replicateM 5 (measure hq)
    putStrLn "Multiple measurements (showing randomness):"
    mapM_ (\m -> putStrLn $ "  " ++ show m) measurements
    
    -- Create distribution from measurement
    let avg_measurement = Float (sum (map extractFloat measurements) / 5.0)
    let measurement_dist = distribution avg_measurement
    putStrLn $ "Measurement distribution: " ++ show measurement_dist

-- | Quantum circuit with uncertainty
quantumCircuitWithUncertainty :: IO ()
quantumCircuitWithUncertainty = do
    putStrLn "\n=== Quantum Circuit with Uncertainty ==="
    
    -- Step 1: Quantum preparation
    let q1 = qubit 0.3
    let q2 = qubit 0.7
    
    -- Step 2: Quantum operations
    let hq1 = hadamard q1
    let hq2 = hadamard q2
    
    -- Step 3: Measurements
    m1 <- measure hq1
    m2 <- measure hq2
    
    putStrLn $ "Measurement 1: " ++ show m1
    putStrLn $ "Measurement 2: " ++ show m2
    
    -- Step 4: Create probabilistic model
    let combined = tensor [extractFloat m1, extractFloat m2]
    let combined_dist = distribution combined
    
    -- Step 5: Sample from the distribution
    sample1 <- sample combined_dist
    sample2 <- sample combined_dist
    
    putStrLn $ "Probabilistic sample 1: " ++ show sample1
    putStrLn $ "Probabilistic sample 2: " ++ show sample2

-- | Bayesian quantum inference
bayesianQuantumInference :: IO ()
bayesianQuantumInference = do
    putStrLn "\n=== Bayesian Quantum Inference ==="
    
    -- Prior belief (quantum state)
    let prior_q = qubit 0.5
    putStrLn $ "Prior quantum state: " ++ show prior_q
    
    -- Evidence (measurement)
    evidence <- measure prior_q
    putStrLn $ "Evidence (measurement): " ++ show evidence
    
    -- Update belief based on evidence
    let updated_q = qubit (extractFloat evidence * 0.8)
    putStrLn $ "Updated quantum state: " ++ show updated_q
    
    -- Create posterior distribution
    let posterior_dist = distribution updated_q
    putStrLn $ "Posterior distribution: " ++ show posterior_dist
    
    -- Sample from posterior
    posterior_sample <- sample posterior_dist
    putStrLn $ "Posterior sample: " ++ show posterior_sample

-- | Helper functions (simplified versions from main Feyn.hs)
data FeynValue = Float Double | Qubit (Double, Double) | Tensor [Double] | Distribution FeynValue deriving (Show)

qubit :: Double -> FeynValue
qubit theta = Qubit (cos theta, sin theta)

hadamard :: FeynValue -> FeynValue
hadamard (Qubit (a, b)) = Qubit ((a + b) / sqrt 2.0, (a - b) / sqrt 2.0)
hadamard v = error $ "Hadamard: expected qubit, got " ++ show v

measure :: FeynValue -> IO FeynValue
measure (Qubit (a, b)) = do
    let prob0 = a * a
    return $ Float (if prob0 > 0.5 then 0.0 else 1.0)
measure v = error $ "Measure: expected qubit, got " ++ show v

tensor :: [Double] -> FeynValue
tensor values = Tensor values

distribution :: FeynValue -> FeynValue
distribution value = Distribution value

sample :: FeynValue -> IO FeynValue
sample (Distribution value) = return value
sample v = error $ "Sample: expected distribution, got " ++ show v

extractFloat :: FeynValue -> Double
extractFloat (Float x) = x
extractFloat _ = error "Expected Float value"

replicateM :: Int -> IO a -> IO [a]
replicateM n action = sequence (replicate n action)

main :: IO ()
main = do
    putStrLn "Probabilistic Quantum Circuits with Feyn Language"
    putStrLn "================================================="
    
    probabilisticMeasurement
    quantumCircuitWithUncertainty
    bayesianQuantumInference
    
    putStrLn "\nThis demonstrates how quantum measurements naturally"
    putStrLn "lead to probabilistic models, enabling uncertainty"
    putStrLn "quantification in quantum computations."
