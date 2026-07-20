
load('./data/UCI_Credit_Card.RData')

context("Testy funkcji dopasowanie_do_zmiennej")

model2 <- glm(default.payment.next.month ~ AGE + I(AGE^2) + BILL_AMT1, 
             data = UCI_Credit_Card, family = binomial)

dop <- dopasowanie_do_zmiennej(UCI_Credit_Card$AGE, predict(model2, type='response'), bckt)

wyn_test <- c(0.2425749, 0.2349437, 0.2287086, 0.2155576,
              0.2114223, 0.2377958, 0.3211251, 0.2212000
  )

wyn_test2 <- as.numeric(round(dop$model, 7))

print (wyn_test- wyn_test2)
#test_that("dopasowanie_do_zmiennej test średnich predyckji na buckecie", {
#  expect_equal(wyn_test, wyn_test2)
#})
