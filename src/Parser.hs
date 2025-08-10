{-# LANGUAGE OverloadedStrings #-}

module Parser
  ( parseProgram
  , parseExprText
  ) where

import           Prelude hiding (lex)
import           Data.Void (Void)
import           Data.Text (Text)
import qualified Data.Text as T
import           Data.Char (isAlphaNum, isLetter)

import           Text.Megaparsec
import           Text.Megaparsec.Char
import qualified Text.Megaparsec.Char.Lexer as L
import qualified Control.Monad.Combinators.Expr as E

import           Feyn

type Parser = Parsec Void Text

sc :: Parser ()
sc = L.space space1 (L.skipLineComment "--") (L.skipBlockComment "/*" "*/")

symbol :: Text -> Parser Text
symbol = L.symbol sc

lexeme :: Parser a -> Parser a
lexeme = L.lexeme sc

parens :: Parser a -> Parser a
parens = between (symbol "(") (symbol ")")

brackets :: Parser a -> Parser a
brackets = between (symbol "[") (symbol "]")

comma :: Parser Text
comma = symbol ","

semi :: Parser Text
semi = symbol ";"

identifier :: Parser Text
identifier = lexeme $ do
  first <- satisfy (\c -> isLetter c || c == '_')
  rest <- takeWhileP (Just "ident") (\c -> isAlphaNum c || c == '_' || c == '-')
  return (T.cons first rest)

reserved :: [Text]
reserved =
  [ "let","in","if","then","else","true","false"
  , "print","tensor","matrix","vector","dist","uniform","normal","bernoulli"
  , "sample","gradient","dot","hadamard","measure","qubit"
  ]

identNotReserved :: Parser Text
identNotReserved = try $ do
  x <- identifier
  if x `elem` reserved then fail "reserved" else return x

-- literals
pBool :: Parser Expr
pBool = (Lit (VBool True) <$ symbol "true") <|> (Lit (VBool False) <$ symbol "false")

pFloat :: Parser Expr
pFloat = do
  n <- lexeme $ L.signed sc (try L.float <|> (fromInteger <$> L.decimal))
  return (Lit (VFloat n))

pString :: Parser Expr
pString = do
  _ <- char '"'
  chars <- manyTill L.charLiteral (char '"')
  sc
  return (Lit (VString (T.pack chars)))

pVar :: Parser Expr
pVar = Var <$> identNotReserved

-- function calls and special forms
pCall1 :: Text -> (Expr -> Expr) -> Parser Expr
pCall1 name ctor = do
  _ <- symbol name
  e <- parens pExpr
  return (ctor e)

pCall2 :: Text -> (Expr -> Expr -> Expr) -> Parser Expr
pCall2 name ctor = do
  _ <- symbol name
  (e1, e2) <- parens $ do
    a <- pExpr
    _ <- comma
    b <- pExpr
    return (a,b)
  return (ctor e1 e2)

pTensor :: Parser Expr
pTensor = do
  _ <- symbol "tensor"
  es <- brackets (pExpr `sepBy` comma)
  return (Tensor es)

pVector :: Parser Expr
pVector = do
  _ <- symbol "vector"
  es <- brackets (pExpr `sepBy` comma)
  return (Vector es)

-- control structures
pIf :: Parser Expr
pIf = do
  _ <- symbol "if"
  c <- pExpr
  _ <- symbol "then"
  t <- pExpr
  _ <- symbol "else"
  e <- pExpr
  return (If c t e)

pLet :: Parser Expr
pLet = do
  _ <- symbol "let"
  name <- identNotReserved
  _ <- symbol "="
  v <- pExpr
  _ <- symbol "in"
  body <- pExpr
  return (Let name v body)

pPrint :: Parser Expr
pPrint = do
  _ <- symbol "print"
  e <- parens pExpr
  return (Print e)

-- primary terms
pTerm :: Parser Expr
pTerm = choice
  [ pIf
  , pLet
  , pPrint
  , pBool
  , try pString
  , pFloat
  , try (pCall1 "qubit" Qubit)
  , try (pCall1 "hadamard" Hadamard)
  , try (pCall2 "rotation" Rotation)
  , try (pCall1 "measure" Measure)
  , try (pCall1 "dist" Dist)
  , try (pCall2 "uniform" Uniform)
  , try (pCall2 "normal" Normal)
  , try (pCall1 "bernoulli" Bernoulli)
  , try (pCall1 "sample" Sample)
  , try (pCall1 "gradient" Gradient)
  , try (pCall1 "mean" Mean)
  , try (pCall2 "mse" MSE)
  , try (pCall2 "grad_mse" GradMSE)
  , try (pCall2 "dot" Dot)
  , try pTensor
  , try pVector
  , Var <$> identNotReserved
  , parens pExpr
  ]

-- expression operators
-- precedence from low to high: or, and, comparisons, +-, */, pow, unary

-- expression operators

table :: [[E.Operator Parser Expr]]
table =
  [ [ prefix "not" Not ]
  , [ binary "^" Pow E.InfixR ]
  , [ binary "*" Mul E.InfixL
    , binary "/" Div E.InfixL
    ]
  , [ binary "+" Add E.InfixL
    , binary "-" Sub E.InfixL
    ]
  , [ binary "==" Eq E.InfixN
    , binary "!=" Neq E.InfixN
    , binary "<=" Lte E.InfixN
    , binary ">=" Gte E.InfixN
    , binary "<" Lt E.InfixN
    , binary ">" Gt E.InfixN
    ]
  , [ binary "and" And E.InfixL
    , binary "or" Or E.InfixL
    ]
  ]
  where
    binary s f assoc = assoc (f <$ symbol s)
    prefix s f       = E.Prefix (f <$ symbol s)

pExpr :: Parser Expr
pExpr = E.makeExprParser pTerm table

-- program: expressions separated by semicolons; trailing semicolon ok
pProgram :: Parser [Expr]
pProgram = between sc eof (pExpr `sepEndBy` semi)

parseProgram :: Text -> Either String [Expr]
parseProgram t = case runParser pProgram "<feyn>" t of
  Left e  -> Left (errorBundlePretty e)
  Right v -> Right v

parseExprText :: Text -> Either String Expr
parseExprText t = case runParser (between sc eof pExpr) "<feyn>" t of
  Left e  -> Left (errorBundlePretty e)
  Right v -> Right v


