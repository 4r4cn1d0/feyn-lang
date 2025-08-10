{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators, FlexibleContexts, OverloadedStrings, RankNTypes #-}

module Feyn where

import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Vector as V
import qualified Data.Map as M
import Control.Monad.Except
import Control.Monad.State
import System.Random
import Data.Maybe (fromMaybe, isJust, fromJust)
import Data.Complex

-- ============================================================================
-- CORE TYPE SYSTEM
-- ============================================================================

-- | Core types for the Feyn language
data FeynType = 
    TFloat
    | TBool
    | TInt
    | TString
    | TUnit
    | TQubit
    | TTensor
    | TMatrix
    | TDist
    | TQuantumTensor
    deriving (Show, Eq)

-- | Runtime values
data FeynValue = 
    VFloat Double
    | VBool Bool
    | VInt Integer
    | VString Text
    | VUnit
    | VQubit (Double, Double)  -- (|0⟩ amplitude, |1⟩ amplitude), real-only
    | VTwoQubit (Double, Double, Double, Double) -- amplitudes for |00>,|01>,|10>,|11>, real-only
    | VTensor (V.Vector Double)
    | VMatrix (V.Vector (V.Vector Double))
    | VDist Distribution
    | VQuantumTensor (V.Vector Double)
    deriving (Show, Eq)

-- | Environment for variable bindings
type Env = M.Map Text FeynValue

-- ============================================================================
-- EXPRESSION AST
-- ============================================================================

-- | Abstract Syntax Tree for Feyn Language
data Expr = 
    -- Literals
    Lit FeynValue
    
    -- Variables
    | Var Text
    
    -- Arithmetic operations
    | Add Expr Expr
    | Sub Expr Expr
    | Mul Expr Expr
    | Div Expr Expr
    | Pow Expr Expr
    | Neg Expr
    
    -- Comparison operations
    | Eq Expr Expr
    | Neq Expr Expr
    | Lt Expr Expr
    | Gt Expr Expr
    | Lte Expr Expr
    | Gte Expr Expr
    
    -- Logical operations
    | And Expr Expr
    | Or Expr Expr
    | Not Expr
    
    -- Quantum operations
    | Qubit Expr
    | Hadamard Expr
    | CNOT Expr Expr
    | Phase Expr Expr
    | Rotation Expr Expr
    | Measure Expr
    | Entangle Expr Expr
    
    -- Differentiable operations
    | Tensor [Expr]
    | Matrix [[Expr]]
    | Vector [Expr]
    | Dot Expr Expr
    | Cross Expr Expr
    | MatMul Expr Expr
    | Transpose Expr
    | Determinant Expr
    | Inverse Expr
    | Eigenvalues Expr
    | Gradient Expr
    | Hessian Expr
    | Jacobian Expr
    | Mean Expr
    | MSE Expr Expr
    | GradMSE Expr Expr
    
    -- Probabilistic operations
    | Dist Expr
    | Normal Expr Expr
    | Uniform Expr Expr
    | Bernoulli Expr
    | Sample Expr
    | Probability Expr
    | Entropy Expr
    | KLDivergence Expr Expr
    | Prior Expr
    | Posterior Expr Expr
    
    -- Hybrid operations
    | QuantumGradient Expr
    | ProbabilisticQuantum Expr
    | DifferentiableDistribution Expr
    
    -- Control flow
    | If Expr Expr Expr
    | Let Text Expr Expr
    
    -- I/O operations
    | Print Expr
    
    -- Error handling
    | Try Expr Expr
    deriving (Show)

-- ============================================================================
-- INTERPRETER
-- ============================================================================

-- | Evaluation error type
data EvalError = 
    UndefinedVariable Text
    | TypeMismatch String
    | RuntimeError String
    | QuantumError String
    | DifferentiableError String
    | ProbabilisticError String
    | DivisionByZero
    | InvalidOperation String
    deriving (Show, Eq)

-- | Evaluation context
data EvalContext = EvalContext
    { env :: Env
    , randomGen :: StdGen
    , quantumState :: QuantumState
    , differentiableState :: DifferentiableState
    , probabilisticState :: ProbabilisticState
    }

-- | Quantum state
data QuantumState = QuantumState
    { qubits :: V.Vector (Double, Double)
    , measurements :: [Double]
    }

-- | Differentiable state
data DifferentiableState = DifferentiableState
    { tensors :: V.Vector (V.Vector Double)
    , gradients :: V.Vector (V.Vector Double)
    }

-- | Probabilistic state
data ProbabilisticState = ProbabilisticState
    { distributions :: [Distribution]
    , samples :: [Double]
    }

-- | Distribution type
data Distribution = 
    NormalDist Double Double
    | UniformDist Double Double
    | BernoulliDist Double
    | Deterministic Double
    deriving (Show, Eq)

-- | Evaluation monad
type EvalM = ExceptT EvalError (StateT EvalContext IO)

-- ============================================================================
-- EVALUATION FUNCTIONS
-- ============================================================================

-- | Main evaluation function
eval :: Expr -> IO (Either EvalError FeynValue)
eval expr = evalStateT (runExceptT (evalExpr expr)) initialContext

-- | Evaluate with environment
evalWithEnv :: Expr -> Env -> IO (Either EvalError FeynValue)
evalWithEnv expr env = evalStateT (runExceptT (evalExpr expr)) (initialContext { env = env })

-- | Evaluate expression in monad
evalExpr :: Expr -> EvalM FeynValue
evalExpr expr = case expr of
    -- Literals
    Lit v -> return v
    
    -- Variables
    Var name -> do
        env <- gets env
        case M.lookup name env of
            Just v -> return v
            Nothing -> throwError $ UndefinedVariable name
    
    -- Arithmetic operations
    Add e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VFloat x, VFloat y) -> return $ VFloat (x + y)
            (VTensor a, VTensor b) ->
                if V.length a == V.length b
                    then return $ VTensor (V.zipWith (+) a b)
                    else throwError $ RuntimeError "Tensor dimensions don't match for addition"
            (VFloat x, VTensor b) -> return $ VTensor (V.map (+ x) b)
            (VTensor a, VFloat y) -> return $ VTensor (V.map (+ y) a)
            _ -> throwError $ TypeMismatch "Add: expected Float or Tensor values"
    
    Sub e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VFloat x, VFloat y) -> return $ VFloat (x - y)
            (VTensor a, VTensor b) ->
                if V.length a == V.length b
                    then return $ VTensor (V.zipWith (-) a b)
                    else throwError $ RuntimeError "Tensor dimensions don't match for subtraction"
            (VFloat x, VTensor b) -> return $ VTensor (V.map (x -) b)
            (VTensor a, VFloat y) -> return $ VTensor (V.map (\x' -> x' - y) a)
            _ -> throwError $ TypeMismatch "Sub: expected Float or Tensor values"
    
    Mul e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VFloat x, VFloat y) -> return $ VFloat (x * y)
            (VTensor a, VTensor b) ->
                if V.length a == V.length b
                    then return $ VTensor (V.zipWith (*) a b)
                    else throwError $ RuntimeError "Tensor dimensions don't match for multiplication"
            (VFloat x, VTensor b) -> return $ VTensor (V.map (x *) b)
            (VTensor a, VFloat y) -> return $ VTensor (V.map (* y) a)
            _ -> throwError $ TypeMismatch "Mul: expected Float or Tensor values"
    
    Div e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VFloat x, VFloat y) -> 
                if y == 0 
                    then throwError DivisionByZero
                    else return $ VFloat (x / y)
            (VTensor a, VTensor b) ->
                if V.length a == V.length b
                    then if V.any (== 0) b then throwError DivisionByZero else return (VTensor (V.zipWith (/) a b))
                    else throwError $ RuntimeError "Tensor dimensions don't match for division"
            (VFloat x, VTensor b) -> if V.any (== 0) b then throwError DivisionByZero else return (VTensor (V.map (x /) b))
            (VTensor a, VFloat y) -> if y == 0 then throwError DivisionByZero else return (VTensor (V.map (/ y) a))
            _ -> throwError $ TypeMismatch "Div: expected Float or Tensor values"
    
    Pow e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VFloat x, VFloat y) -> return $ VFloat (x ** y)
            (VTensor a, VFloat y) -> return $ VTensor (V.map (** y) a)
            (VTensor a, VTensor b) ->
                if V.length a == V.length b
                    then return $ VTensor (V.zipWith (**) a b)
                    else throwError $ RuntimeError "Tensor dimensions don't match for pow"
            _ -> throwError $ TypeMismatch "Pow: expected Float or Tensor values"
    
    Neg e -> do
        v <- evalExpr e
        case v of
            VFloat x -> return $ VFloat (-x)
            _ -> throwError $ TypeMismatch "Neg: expected Float value"
    
    -- Comparison operations
    Eq e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        return $ VBool (v1 == v2)
    
    Neq e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        return $ VBool (v1 /= v2)
    
    Lt e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VFloat x, VFloat y) -> return $ VBool (x < y)
            _ -> throwError $ TypeMismatch "Lt: expected Float values"
    
    Gt e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VFloat x, VFloat y) -> return $ VBool (x > y)
            _ -> throwError $ TypeMismatch "Gt: expected Float values"
    
    Lte e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VFloat x, VFloat y) -> return $ VBool (x <= y)
            _ -> throwError $ TypeMismatch "Lte: expected Float values"
    
    Gte e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VFloat x, VFloat y) -> return $ VBool (x >= y)
            _ -> throwError $ TypeMismatch "Gte: expected Float values"
    
    -- Logical operations
    And e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VBool x, VBool y) -> return $ VBool (x && y)
            _ -> throwError $ TypeMismatch "And: expected Bool values"
    
    Or e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VBool x, VBool y) -> return $ VBool (x || y)
            _ -> throwError $ TypeMismatch "Or: expected Bool values"
    
    Not e -> do
        v <- evalExpr e
        case v of
            VBool x -> return $ VBool (not x)
            _ -> throwError $ TypeMismatch "Not: expected Bool value"
    
    -- Quantum operations
    Qubit e -> do
        v <- evalExpr e
        case v of
            VFloat theta -> do
                let cosTheta = cos theta
                let sinTheta = sin theta
                return $ VQubit (cosTheta, sinTheta)
            _ -> throwError $ TypeMismatch "Qubit: expected Float value"
    
    Hadamard e -> do
        v <- evalExpr e
        case v of
            VQubit (a, b) -> do
                let sqrt2 = sqrt 2.0
                let newA = (a + b) / sqrt2
                let newB = (a - b) / sqrt2
                return $ VQubit (newA, newB)
            _ -> throwError $ TypeMismatch "Hadamard: expected Qubit value"
    
    CNOT control target -> do
        c <- evalExpr control
        t <- evalExpr target
        case (c, t) of
            (VQubit (aC, bC), VQubit (t0, t1)) -> do
                -- Probabilistic CNOT via measuring control; avoids entanglement in this simplified model
                let p1 = bC * bC
                gen <- gets randomGen
                let (r, newGen) = randomR (0.0, 1.0) gen
                modify (\ctx -> ctx { randomGen = newGen })
                let flipTarget = r < p1
                let newT0 = if flipTarget then t1 else t0
                let newT1 = if flipTarget then t0 else t1
                return $ VQubit (newT0, newT1)
            _ -> throwError $ TypeMismatch "CNOT: expected Qubit values"
    
    Phase angle e -> do
        -- In a real-amplitude simulator, a general phase does not affect measurement probabilities.
        -- We therefore return the input state unchanged to represent a Z-phase with no probability effect.
        _ <- evalExpr angle
        v <- evalExpr e
        case v of
            VQubit _ -> return v
            _ -> throwError $ TypeMismatch "Phase: expected Qubit"
    
    Rotation angle e -> do
        a <- evalExpr angle
        q <- evalExpr e
        case (a, q) of
            (VFloat theta, VQubit (q0, q1)) -> do
                let cosTheta = cos theta
                let sinTheta = sin theta
                let newQ0 = q0 * cosTheta - q1 * sinTheta
                let newQ1 = q0 * sinTheta + q1 * cosTheta
                return $ VQubit (newQ0, newQ1)
            _ -> throwError $ TypeMismatch "Rotation: expected Float angle and Qubit"
    
    Measure e -> do
        v <- evalExpr e
        case v of
            VQubit (a, b) -> do
                let prob0 = a * a
                let prob1 = b * b
                gen <- gets randomGen
                let (r, newGen) = randomR (0.0, 1.0) gen
                modify (\ctx -> ctx { randomGen = newGen })
                return $ VFloat (if r < prob0 then 0.0 else 1.0)
            VTwoQubit (a00,a01,a10,a11) -> do
                -- Measure only the first qubit; return 0.0 or 1.0
                let p0 = a00*a00 + a01*a01
                gen <- gets randomGen
                let (r, newGen) = randomR (0.0, 1.0) gen
                modify (\ctx -> ctx { randomGen = newGen })
                return $ VFloat (if r < p0 then 0.0 else 1.0)
            _ -> throwError $ TypeMismatch "Measure: expected Qubit value"
    
    Entangle e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VQubit (a, b), VQubit (c, d)) -> do
                -- Apply H on first, then CNOT(control=first,target=second)
                let h0 = (a + b) / sqrt 2.0
                let h1 = (a - b) / sqrt 2.0
                -- Tensor product |ψ> = [h0*c, h0*d, h1*c, h1*d]
                let t00 = h0 * c
                let t01 = h0 * d
                let t10 = h1 * c
                let t11 = h1 * d
                -- CNOT swaps |10> and |11> components
                let e00 = t00
                let e01 = t01
                let e10 = t11
                let e11 = t10
                return $ VTwoQubit (e00, e01, e10, e11)
            _ -> throwError $ TypeMismatch "Entangle: expected Qubit values"
    
    -- Differentiable operations
    Tensor es -> do
        values <- mapM evalExpr es
        let floatValues = map extractFloat values
        if all isJust floatValues
            then return $ VTensor (V.fromList (map fromJust floatValues))
            else throwError $ TypeMismatch "Tensor: expected Float values"
    
    Matrix rows -> do
        matrixValues <- mapM (mapM evalExpr) rows
        let floatMatrix = map (map extractFloat) matrixValues
        if all (all isJust) floatMatrix
            then return $ VMatrix (V.fromList (map (V.fromList . map fromJust) floatMatrix))
            else throwError $ TypeMismatch "Matrix: expected Float values"
    
    Vector es -> do
        values <- mapM evalExpr es
        let floatValues = map extractFloat values
        if all isJust floatValues
            then return $ VTensor (V.fromList (map fromJust floatValues))
            else throwError $ TypeMismatch "Vector: expected Float values"
    
    Gradient e -> do
        v <- evalExpr e
        case v of
            VTensor values -> do
                let epsilon = 1e-6
                let gradients = V.map (\x -> (x + epsilon - x) / epsilon) values
                return $ VTensor gradients
            _ -> throwError $ TypeMismatch "Gradient: expected Tensor value"

    Mean e -> do
        v <- evalExpr e
        case v of
            VTensor t | V.length t > 0 -> return $ VFloat (V.sum t / fromIntegral (V.length t))
            VTensor _ -> throwError $ RuntimeError "Mean of empty tensor"
            _ -> throwError $ TypeMismatch "Mean: expected Tensor value"

    MSE a b -> do
        va <- evalExpr a
        vb <- evalExpr b
        case (va, vb) of
            (VTensor ta, VTensor tb) -> do
                if V.length ta /= V.length tb
                    then throwError $ RuntimeError "MSE: Tensor dimensions must match"
                    else do
                        let n = fromIntegral (V.length ta)
                            diffs = V.zipWith (-) ta tb
                            loss = V.sum (V.map (\d -> d*d) diffs) / n
                        return (VFloat loss)
            _ -> throwError $ TypeMismatch "MSE: expected Tensor values"

    GradMSE a b -> do
        va <- evalExpr a
        vb <- evalExpr b
        case (va, vb) of
            (VTensor ta, VTensor tb) -> do
                if V.length ta /= V.length tb
                    then throwError $ RuntimeError "GradMSE: Tensor dimensions must match"
                    else do
                        let n = fromIntegral (V.length ta)
                            grads = V.map (\d -> (2.0 / n) * d) (V.zipWith (-) ta tb)
                        return (VTensor grads)
            _ -> throwError $ TypeMismatch "GradMSE: expected Tensor values"

    Mean e -> do
        v <- evalExpr e
        case v of
            VTensor t | V.length t > 0 -> return $ VFloat (V.sum t / fromIntegral (V.length t))
            VTensor _ -> throwError $ RuntimeError "Mean of empty tensor"
            _ -> throwError $ TypeMismatch "Mean: expected Tensor value"
    
    Dot e1 e2 -> do
        v1 <- evalExpr e1
        v2 <- evalExpr e2
        case (v1, v2) of
            (VTensor t1, VTensor t2) -> do
                if V.length t1 == V.length t2
                    then do
                        let dotProduct = V.sum (V.zipWith (*) t1 t2)
                        return $ VFloat dotProduct
                    else throwError $ RuntimeError "Tensor dimensions don't match for dot product"
            _ -> throwError $ TypeMismatch "Dot: expected Tensor values"
    
    -- Probabilistic operations
    Dist e -> do
        v <- evalExpr e
        case v of
            VFloat x -> return $ VDist (Deterministic x)
            _ -> throwError $ TypeMismatch "Dist: expected Float value"
    
    Normal mu sigma -> do
        m <- evalExpr mu
        s <- evalExpr sigma
        case (m, s) of
            (VFloat mean, VFloat std) -> do
                let dist = NormalDist mean std
                modify (\ctx -> ctx { probabilisticState = (probabilisticState ctx) { distributions = dist : distributions (probabilisticState ctx) } })
                return $ VDist dist
            _ -> throwError $ TypeMismatch "Normal: expected Float values"
    
    Uniform min max -> do
        mn <- evalExpr min
        mx <- evalExpr max
        case (mn, mx) of
            (VFloat minVal, VFloat maxVal) -> do
                let dist = UniformDist minVal maxVal
                modify (\ctx -> ctx { probabilisticState = (probabilisticState ctx) { distributions = dist : distributions (probabilisticState ctx) } })
                return $ VDist dist
            _ -> throwError $ TypeMismatch "Uniform: expected Float values"
    
    Bernoulli p -> do
        prob <- evalExpr p
        case prob of
            VFloat probVal -> do
                let dist = BernoulliDist probVal
                modify (\ctx -> ctx { probabilisticState = (probabilisticState ctx) { distributions = dist : distributions (probabilisticState ctx) } })
                return $ VDist dist
            _ -> throwError $ TypeMismatch "Bernoulli: expected Float value"
    
    Sample e -> do
        v <- evalExpr e
        case v of
            VDist dist -> do
                gen <- gets randomGen
                let (r1, gen1) = randomR (0.0, 1.0) gen
                let (r2, gen2) = randomR (0.0, 1.0) gen1
                modify (\ctx -> ctx { randomGen = gen2 })
                case dist of
                    NormalDist mean std -> return $ VFloat (mean + std * boxMuller r1 r2)
                    UniformDist a b -> return $ VFloat (a + r1 * (b - a))
                    BernoulliDist p -> return $ VBool (r1 < p)
                    Deterministic x -> return $ VFloat (x + 0.1 * (r1 - 0.5)) -- small noise for demo parity
            _ -> throwError $ TypeMismatch "Sample: expected distribution"
    
    -- Control flow
    If cond thenExpr elseExpr -> do
        v <- evalExpr cond
        case v of
            VBool True -> evalExpr thenExpr
            VBool False -> evalExpr elseExpr
            _ -> throwError $ TypeMismatch "If: expected Bool condition"
    
    Let name valueExpr bodyExpr -> do
        value <- evalExpr valueExpr
        prevEnv <- gets env
        modify (\ctx -> ctx { env = M.insert name value (env ctx) })
        result <- evalExpr bodyExpr
        modify (\ctx -> ctx { env = prevEnv })
        return result
    
    -- I/O operations
    Print e -> do
        v <- evalExpr e
        liftIO $ putStrLn $ "Output: " ++ show v
        return VUnit
    
    -- Error handling
    Try expr handler -> do
        result <- catchError (evalExpr expr) (\_ -> evalExpr handler)
        return result
    
    -- Placeholder implementations for other expressions
    _ -> throwError $ RuntimeError "Expression not yet implemented"

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- | Initial evaluation context
initialContext :: EvalContext
initialContext = EvalContext
    { env = M.empty
    , randomGen = mkStdGen 42
    , quantumState = QuantumState V.empty []
    , differentiableState = DifferentiableState V.empty V.empty
    , probabilisticState = ProbabilisticState [] []
    }

-- | Extract float value
extractFloat :: FeynValue -> Maybe Double
extractFloat (VFloat x) = Just x
extractFloat _ = Nothing

-- | Extract tensor values
extractTensor :: FeynValue -> Maybe (V.Vector Double)
extractTensor (VTensor vec) = Just vec
extractTensor _ = Nothing

-- | Extract qubit values
extractQubit :: FeynValue -> Maybe (Double, Double)
extractQubit (VQubit q) = Just q
extractQubit _ = Nothing

-- | Complex number helper (using Data.Complex)

-- ============================================================================
-- CONVENIENCE FUNCTIONS
-- ============================================================================

-- | Create a qubit
qubit :: Double -> Expr
qubit angle = Qubit (Lit (VFloat angle))

-- | Apply Hadamard gate
hadamard :: Expr -> Expr
hadamard = Hadamard

-- | Measure a qubit
measure :: Expr -> Expr
measure = Measure

-- | Create a tensor
tensor :: [Double] -> Expr
tensor values = Tensor (map (Lit . VFloat) values)

-- | Compute gradient
grad :: Expr -> Expr
grad = Gradient

-- | Dot product
dot :: Expr -> Expr -> Expr
dot = Dot

-- | Create a distribution
dist :: Expr -> Expr
dist = Dist

-- | Sample from distribution
sample :: Expr -> Expr
sample = Sample

-- | Run a Feyn program
runFeyn :: Expr -> IO (Either EvalError FeynValue)
runFeyn = eval

-- | Run a Feyn program with environment
runFeynWithEnv :: Expr -> Env -> IO (Either EvalError FeynValue)
runFeynWithEnv = evalWithEnv

-- ============================================================================
-- INTERNAL MATH HELPERS
-- ============================================================================

-- | Box-Muller transform to sample standard normal from two uniforms in (0,1)
boxMuller :: Double -> Double -> Double
boxMuller u1 u2 =
    let epsilon = 1e-12
        u1' = if u1 <= 0 then epsilon else if u1 >= 1 then 1 - epsilon else u1
        r = sqrt (-2.0 * log u1')
        theta = 2.0 * pi * u2
    in r * cos theta
