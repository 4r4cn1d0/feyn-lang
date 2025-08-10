import System.CPUTime
import Data.Time.Clock.POSIX (getPOSIXTime)
import Data.List (foldl')
import Text.Printf

{-|
 Monte Carlo π calculation using random sampling.
-}
monteCarloPi :: Int -> IO (Double, Double, Double)
monteCarloPi samples = do
    startT <- getPOSIXTime
    let seed = (floor (startT * 1000)) `mod` 2147483647 :: Int
        xs = take (2 * samples) (lcgDoubles seed)
        pairs = toPairs xs
        insideCircle = foldl' (\acc (x, y) -> if x * x + y * y <= 1.0 then acc + 1 else acc) 0 (take samples pairs)
        piEstimate = 4.0 * fromIntegral insideCircle / fromIntegral samples
        err = abs (piEstimate - pi)
    endT <- getPOSIXTime
    let executionTime = realToFrac (endT - startT) :: Double
    return (piEstimate, err, executionTime)
  where
    toPairs :: [Double] -> [(Double, Double)]
    toPairs (a:b:rest) = (a, b) : toPairs rest
    toPairs _ = []

{-|
 Simple linear congruential generator to avoid external dependencies.
-}
lcgDoubles :: Int -> [Double]
lcgDoubles seed =
    let m = 2147483647 :: Int
        a = 48271 :: Int
        seqInts = tail $ iterate (\x -> (a * x) `mod` m) (seed `mod` m)
    in map (\x -> fromIntegral x / fromIntegral m) seqInts

-- Simplified Neural Network Training
neuralNetworkTraining :: Int -> IO (Double, Double, [Double])
neuralNetworkTraining iterations = do
    startTime <- getCPUTime
    
    -- Training data: y = 2x + 1 + noise
    let xData = [1, 2, 3, 4, 5]
        yData = [3.1, 5.2, 7.1, 9.3, 11.2]
        learningRate = 0.01
        
        -- Forward pass
        forward w b x = w * x + b
        
        -- Loss function (MSE)
        loss w b = sum [(forward w b x - y) ^ 2 | (x, y) <- zip xData yData] / fromIntegral (length xData)
        
        -- Gradient calculation
        gradients w b = let yPreds = map (forward w b) xData
                            errors = zipWith (-) yPreds yData
                            dw = 2 * sum (zipWith (*) errors xData) / fromIntegral (length xData)
                            db = 2 * sum errors / fromIntegral (length xData)
                        in (dw, db)
        
        -- Training loop
        trainStep (w, b) = let (dw, db) = gradients w b
                           in (w - learningRate * dw, b - learningRate * db)
        
        -- Run training
        initialWeights = (0.0, 0.0)
        (finalW, finalB, lossHistory) = foldl' (\(w, b, history) i -> 
            let (newW, newB) = trainStep (w, b)
                currentLoss = loss newW newB
            in (newW, newB, history ++ [currentLoss])) (0.0, 0.0, []) [1..iterations]
    
    endTime <- getCPUTime
    let executionTime = fromIntegral (endTime - startTime) / 1e12
        finalLoss = last lossHistory
    
    return (finalLoss, executionTime, lossHistory)

-- Benchmark runner
runHaskellBenchmarks :: IO ()
runHaskellBenchmarks = do
    putStrLn "Haskell Benchmarks"
    putStrLn $ replicate 50 '='
    
    -- Monte Carlo Integration
    putStrLn "\nMonte Carlo Integration (pi calculation)"
    putStrLn $ replicate 40 '-'
    
    let sampleSizes = [1000, 10000, 100000, 1000000]
    
    piResults <- mapM (\samples -> do
        let trials = 5
        vals <- sequence [monteCarloPi samples | _ <- [1..trials]]
        let pis = map (\(p,_,_) -> p) vals
            errs = map (\(_,e,_) -> e) vals
            times = map (\(_,_,t) -> t) vals
            avg f = sum f / fromIntegral (length f)
            piEst = avg pis
            err = avg errs
            time = avg times
        printf "Samples: %7d | pi ≈ %.6f | Error: %.6f | Time: %.6fs\n" 
               samples piEst err time
        mapM_ (\(p,e,t) -> printf "MC,%d,%.6f,%.6f,%.9f\n" samples p e t) vals
        return (samples, piEst, err, time)) sampleSizes
    
    -- Neural Network Training
    putStrLn "\nNeural Network Training"
    putStrLn $ replicate 40 '-'
    
    let iterations = 1000
    (finalLoss, time, lossHistory) <- neuralNetworkTraining iterations
    printf "Iterations: %d | Final Loss: %.6f | Time: %.4fs\n" 
           iterations finalLoss time
    -- Machine-readable line for parsers
    printf "NN,%d,%.6f,%.6f\n" iterations finalLoss time
    
    -- Print summary
    putStrLn "\nSummary"
    putStrLn $ replicate 40 '-'
    let (_, piEst1M, err1M, _) = last piResults
    printf "Monte Carlo pi (1M samples): %.6f (Error: %.6f)\n" piEst1M err1M
    printf "Neural Network Final Loss: %.6f\n" finalLoss

main :: IO ()
main = runHaskellBenchmarks
