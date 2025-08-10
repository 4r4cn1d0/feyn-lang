module Main (main) where

import Test.Hspec
import Feyn

main :: IO ()
main = hspec $ do
  describe "Feyn core" $ do
    it "evaluates basic arithmetic" $ do
      res <- eval (Add (Lit (VFloat 2.0)) (Lit (VFloat 3.0)))
      res `shouldBe` Right (VFloat 5.0)

    it "creates tensors" $ do
      res <- eval (Tensor [Lit (VFloat 1.0), Lit (VFloat 2.0)])
      case res of
        Right (VTensor _) -> pure ()
        _ -> expectationFailure "Expected VTensor"

    it "dot product returns scalar float" $ do
      let v1 = Tensor [Lit (VFloat 1.0), Lit (VFloat 2.0), Lit (VFloat 3.0)]
      let v2 = Tensor [Lit (VFloat 4.0), Lit (VFloat 5.0), Lit (VFloat 6.0)]
      res1 <- eval v1
      res2 <- eval v2
      case (res1, res2) of
        (Right t1, Right t2) -> do
          r <- eval (Dot (Lit t1) (Lit t2))
          r `shouldBe` Right (VFloat 32.0)
        _ -> expectationFailure "Vector construction failed"

    it "sampling from distributions works" $ do
      r1 <- eval (Sample (Normal (Lit (VFloat 0.0)) (Lit (VFloat 1.0))))
      r2 <- eval (Sample (Uniform (Lit (VFloat 0.0)) (Lit (VFloat 1.0))))
      r3 <- eval (Sample (Bernoulli (Lit (VFloat 0.7))))
      case (r1, r2, r3) of
        (Right (VFloat _), Right (VFloat _), Right (VBool _)) -> pure ()
        _ -> expectationFailure "Sampling types incorrect"


