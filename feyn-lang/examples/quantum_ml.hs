#!/usr/bin/env runhaskell

-- Quantum Machine Learning Example
-- Demonstrates quantum feature extraction + differentiable optimization

-- Import the main Feyn language
-- (In a real implementation, this would be a proper module import)

-- | Quantum feature extraction
quantumFeatureExtraction :: IO ()
quantumFeatureExtraction = do
    putStrLn "=== Quantum Feature Extraction ==="
    
    -- Create quantum features
    let q1 = qubit 0.3
    let q2 = qubit 0.7
    
    -- Apply quantum gates
    let hq1 = hadamard q1
    let hq2 = hadamard q2
    
    -- Measure quantum features
    m1 <- measure hq1
    m2 <- measure hq2
    
    putStrLn $ "Quantum feature 1: " ++ show m1
    putStrLn $ "Quantum feature 2: " ++ show m2
    
    -- Convert to differentiable tensors
    let features = tensor [extractFloat m1, extractFloat m2]
    putStrLn $ "Feature tensor: " ++ show features
    
    -- Compute gradients for optimization
    let gradients = gradient features
    putStrLn $ "Feature gradients: " ++ show gradients

-- | Extract float value from FeynValue
extractFloat :: FeynValue -> Double
extractFloat (Float x) = x
extractFloat _ = error "Expected Float value"

-- | Simple quantum neural network
quantumNeuralNetwork :: IO ()
quantumNeuralNetwork = do
    putStrLn "\n=== Quantum Neural Network ==="
    
    -- Input layer (quantum)
    let input_q = qubit 0.5
    let input_h = hadamard input_q
    input_m <- measure input_h
    
    -- Hidden layer (differentiable)
    let hidden = tensor [extractFloat input_m, 0.5, 0.3]
    let hidden_grad = gradient hidden
    
    -- Output layer (probabilistic)
    let output_dist = distribution (Float (sum (extractTensor hidden)))
    output_sample <- sample output_dist
    
    putStrLn $ "Input measurement: " ++ show input_m
    putStrLn $ "Hidden layer: " ++ show hidden
    putStrLn $ "Hidden gradients: " ++ show hidden_grad
    putStrLn $ "Output sample: " ++ show output_sample

-- | Extract tensor values from FeynValue
extractTensor :: FeynValue -> [Double]
extractTensor (Tensor values) = values
extractTensor _ = error "Expected Tensor value"

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

gradient :: FeynValue -> FeynValue
gradient (Tensor values) = Tensor (map (\x -> 1.0) values)
gradient v = error $ "Gradient: expected tensor, got " ++ show v

distribution :: FeynValue -> FeynValue
distribution value = Distribution value

sample :: FeynValue -> IO FeynValue
sample (Distribution value) = return value
sample v = error $ "Sample: expected distribution, got " ++ show v

main :: IO ()
main = do
    putStrLn "Quantum Machine Learning with Feyn Language"
    putStrLn "============================================="
    
    quantumFeatureExtraction
    quantumNeuralNetwork
    
    putStrLn "\nThis demonstrates how quantum features can be"
    putStrLn "seamlessly integrated with differentiable optimization"
    putStrLn "and probabilistic modeling in a unified framework."
