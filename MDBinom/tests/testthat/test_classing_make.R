# TODO: Add comment
# 
# Author: Piotr
###############################################################################

library(MDBinom)
library(testthat)

context("Testy dyskretyzacji z wagami")

set.seed(1)

x <- c(rep(1,100),1:100)
y <- rbinom(200, 1, 0.1)
w <- rnorm(200)


x <- c(1)
y <- c(1)
buckety_br(x,y,10)
#bckt_br(x,y,10)




x <- c(1,1,1,1,1)
y <- c(0,0,0,0,0) 
bckt <- MDBinom::bckt_stat(x,y,1)

test_that("bckt_stat czy Total jest ostatnim wierszem", {
  expect_equal(bckt$label[2], 'TOTAL')
})

bckt <- bckt_stat2(breaks = c(0,1), x = x, y = y, weights = 1)
test_that("bckt_stat2 czy Total jest ostatnim wierszem", {
  expect_equal(bckt$label[nrow(bckt)], 'TOTAL')
})

test_that("bckt_stat test typu kolumny discrete", {
  expect_equal(typeof(bckt_stat(x,y,1)$discret), 'double')
  expect_equal(typeof(bckt_stat(as.character(x),y,1)$discret), 'character')
})


#bckt_br z powyższym

set.seed(1)

x <- c(rep(1,100),1:100)
y <- rbinom(200, 1, 0.1)
w <- abs(rnorm(200))
breaks=c(1,50,100)

bckt <- bckt_stat2(breaks,x, y, weights = w)
test_that("bckt_stat2 czy Total jest ostatnim wierszem", {
  expect_equal(bckt$label[nrow(bckt)], 'TOTAL')
})

bckt <- bckt_br(x,y,4,w)
test_that("bckt_br czy Total jest ostatnim wierszem", {
  expect_equal(bckt$label[nrow(bckt)], 'TOTAL')
})

test_that("bckt_br czy poprawnie liczona średnia br na Totalu", {
  expect_equal(bckt$br[nrow(bckt)], sum(y*w)/sum(w))
})


