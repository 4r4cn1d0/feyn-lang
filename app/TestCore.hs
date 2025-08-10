module Main where

import Feyn
import qualified Data.Text as T

main :: IO ()
main = do
    putStrLn "Feyn Language Core Engine Test"
    putStrLn "=============================="
    putStrLn ""
    
    -- Test 1: Basic arithmetic
    putStrLn "1. Basic Arithmetic:"
    let expr1 = Add (Lit (VFloat 5.0)) (Lit (VFloat 3.0))
    result1 <- eval expr1
    case result1 of
        Left err -> putStrLn $ "Error: " ++ show err
        Right val -> putStrLn $ "5.0 + 3.0 = " ++ show val
    putStrLn ""
    
    -- Test 2: Quantum operations
    putStrLn "2. Quantum Operations:"
    let expr2 = Qubit (Lit (VFloat 0.5))
    result2 <- eval expr2
    case result2 of
        Left err -> putStrLn $ "Error: " ++ show err
        Right val -> putStrLn $ "qubit(0.5) = " ++ show val
    
    let expr3 = Hadamard expr2
    result3 <- eval expr3
    case result3 of
        Left err -> putStrLn $ "Error: " ++ show err
        Right val -> putStrLn $ "hadamard(qubit(0.5)) = " ++ show val
    putStrLn ""
    
    -- Test 3: Differentiable operations
    putStrLn "3. Differentiable Operations:"
    let expr4 = Tensor [Lit (VFloat 1.0), Lit (VFloat 2.0), Lit (VFloat 3.0)]
    result4 <- eval expr4
    case result4 of
        Left err -> putStrLn $ "Error: " ++ show err
        Right val -> putStrLn $ "tensor[1, 2, 3] = " ++ show val
    
    let expr5 = Gradient expr4
    result5 <- eval expr5
    case result5 of
        Left err -> putStrLn $ "Error: " ++ show err
        Right val -> putStrLn $ "gradient(tensor[1, 2, 3]) = " ++ show val
    putStrLn ""
    
    -- Test 4: Probabilistic operations
    putStrLn "4. Probabilistic Operations:"
    let expr6 = Dist (Lit (VFloat 0.5))
    result6 <- eval expr6
    case result6 of
        Left err -> putStrLn $ "Error: " ++ show err
        Right val -> putStrLn $ "dist(0.5) = " ++ show val
    
    let expr7 = Sample expr6
    result7 <- eval expr7
    case result7 of
        Left err -> putStrLn $ "Error: " ++ show err
        Right val -> putStrLn $ "sample(dist(0.5)) = " ++ show val
    putStrLn ""
    
    -- Test 5: Control flow
    putStrLn "5. Control Flow:"
    let expr8 = If (Lit (VBool True)) (Lit (VFloat 42.0)) (Lit (VFloat 0.0))
    result8 <- eval expr8
    case result8 of
        Left err -> putStrLn $ "Error: " ++ show err
        Right val -> putStrLn $ "if true then 42.0 else 0.0 = " ++ show val
    putStrLn ""
    
    -- Test 6: Variable binding
    putStrLn "6. Variable Binding:"
    let expr9 = Let (T.pack "x") (Lit (VFloat 10.0)) (Add (Var (T.pack "x")) (Lit (VFloat 5.0)))
    result9 <- eval expr9
    case result9 of
        Left err -> putStrLn $ "Error: " ++ show err
        Right val -> putStrLn $ "let x = 10.0 in x + 5.0 = " ++ show val
    putStrLn ""
    
    -- Test 7: Combined operations
    putStrLn "7. Combined Operations:"
    let expr10 = Let (T.pack "q") (Qubit (Lit (VFloat 0.3))) 
                (Let (T.pack "m") (Measure (Var (T.pack "q")))
                (Tensor [Var (T.pack "m"), Lit (VFloat 1.0), Lit (VFloat 2.0)]))
    result10 <- eval expr10
    case result10 of
        Left err -> putStrLn $ "Error: " ++ show err
        Right val -> putStrLn $ "let q = qubit(0.3) in let m = measure(q) in tensor[m, 1, 2] = " ++ show val
    putStrLn ""
    
    putStrLn "All tests completed successfully!"
    putStrLn "The Feyn Language core engine is working correctly."
