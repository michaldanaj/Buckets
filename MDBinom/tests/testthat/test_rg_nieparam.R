

context("Testy funkcji rg_nieparam")

#bckt_br z powyższym

set.seed(1)

x <- c(rep(1,100),1:100)
y <- rbinom(200, 1, 0.1)
weights <- abs(rnorm(200))
breaks=c(1,50,100)

bckt <- rg_nieparam(score=x, default=y, plot=FALSE, buckets = 3)

test_that("rg_nieparam czy liczba obserwacji z bucketów daje liczbę w Totalu, bez wag", {
  expect_equal(sum(bckt$n_obs), length(x))
})

test_that("rg_nieparam czy średnia z bucketów daje średnią w Totalu, bez wag", {
  expect_equal(sum(bckt$br * bckt$n_obs) / sum(bckt$n_obs), mean(y))
})



bckt <- rg_nieparam(score=x, default=y, plot=FALSE, buckets = 3, weights = weights)

test_that("rg_nieparam czy liczba obserwacji z bucketów daje liczbę w Totalu, przy wykorzystaniu wag", {
  expect_equal(sum(bckt$n_obs), sum(weights))
})

test_that("rg_nieparam czy średnia z bucketów daje średnią w Totalu, przy wykorzystaniu wag", {
  expect_equal(sum(bckt$br * bckt$n_obs) / sum(bckt$n_obs), sum(y * weights) / sum(weights))
})

