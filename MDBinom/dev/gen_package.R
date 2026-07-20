# TODO: Add comment
# 
# Author: Piotr
###############################################################################


library(devtools)
library(roxygen2)

setwd('G:\\Michal\\eclipse - java\\MDbinom')
#create('MDBinom')	

document()

#check(check_dir =".")

install('MDBinom')

setwd('G:\\Michal\\eclipse - java\\MDbinom')
build(pkg = 'MDBinom', path='.', binary = TRUE)
x	


#  testy
devtools::test()


####




setwd('G:\\Michal\\eclipse - java')
shell('R CMD check  MDBinom', intern=TRUE)
#shell('R CMD INSTALL --build MDBinom', intern=TRUE)




