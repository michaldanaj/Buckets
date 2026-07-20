# Co doda?:
# - dodawanie pojedynczych wierszy do klasy
# - po dodaniu wiersza, nale?y okre?li? stan obiektu jako nieprzeliczony
# - funkcje powinny m?c korzysta? tylko z przeliczonej klasy
# - funkcja do przeliczenia klasy na zadanych danych
# - doda? wyliczanie ?rednich z dowolnych kolumn i dodawanie ich do statystyk klasy 
#   (albo rozszerzy? klas? o tak? mo?liwo??). Mi?dzy innymi by?oby to po to,
# aby m?c doda? ?redni? ze score, ?eby m?c wyrysowywa? j? p??niej 
#
#


# 
# N <- 1e5
# x <- rep(1:10, N/10)
# x <- rep(letters[1:10], N/10)
# y <- rep(rbinom(size=1, n=N, prob=0.4))
# ekstra <- rep(rbinom(size=1, n=N, prob=0.4))
# #breaks=c(-Inf, 1,5.5,6,7,8,910)
# weights=rep(1,length(x))
# dt<-data.table(x,y,weights)
# total=TRUE
# n=2
# method="eq_length"
# 
# for(i in 1:30){
#   # buckety_br(x,y, 5)
#   # bckt_br(x,y, n=5)
#   # 
#   # buckety_stat2(breaks,x,y)
#   # bckt_stat2(breaks, x=x,y=y)
#   
#   buckety_stat(x,y)
#   bckt_stat( x=x,y=y)
# }

#' Dzieli na przedzia?y jedn? z dw?ch metod i wylicza na nich bad rate.
#'
#' Jako ?e jest to wersja przej?ciowa, nie ma ?adnych detali.
#' Mo?e poza tym, ?e je?li nie ma obserwacji w jakim? bukcecie, to go usuwa.
#'
#' @param x  Zmienna, kt?r? b?dziemy dyskretyzowa?.  
#' @param y  Zmienna dwumianowa.  
#' @param n  Liczba wynikowych przedzia??w.  
#' @param method  Spos?b podzia?u na przedzia?y. Szczeg??y w sekcji Details.  
#' @param one.value.action Jeszcze nie dzia?a.
#' @param weights Wagi.
#' 
#' @return  
#' Zwraca \code{data.frame}, w kt?rym dla \code{i}-tego wiersza podane s? statystyki
#' dla \code{i}-tego przedzia?u.
#' \item{nr }{Numer przedzia?u.}
#' \item{od }{Pocz?tek przedzia?u.}
#' \item{srodek}{?rodek przedzia?u.}
#' \item{mean}{?rednia z obserwacji w przedziale.}
#' \item{do}{Koniec przedzia?u.}
#' \item{n_default}{Liczba default?w.}
#' \item{n_obs}{Liczba obserwacji.}
#' \item{br}{Bad rate.}
#' \item{logit}{Logit.}
#' \item{probit}{Probit.}
#' \item{var}{Wariancja zmiennej \code{default}.}
#' \item{waga}{Waga jako odwrotno?? warinacji. Gdy \code{n_default=0} to przyjmuje si?,
#'		?e \code{n_default=0.5}}
#' @examples 
#' x<-rnorm(1000);
#' y<-(x+rnorm(1000))<0;
#' buckety_br(x, y, 10);
#' @author Micha? Danaj
#' @export
buckety_br<-function(x, y, n, method=c("eq_length", "eq_count"), one.value.action=c("none","combine"),
		weights=NULL)
{                                         
#	TODO - domy?lna warto?? method
	method<-match.arg(method);
	
	if (is.null(weights))
		weights<-rep(1,length(x))
	
	#dolny kraniec bucketa (wartosc x)
	od=c(1:n);
	#gorny kranieb bucketa (wartosc x)
	do=c(1:n);
	
	if (method=="eq_length")
		granice<-seq(min(x),max(x),length.out=n+1)
	else{
		#w przypadku, gdy jedna warto?? jest dla wielu kwantyli, zdarzaj? si? problemy numeryczne
		#?e warto?? teoretycznie jest taka sama, ale r??ni si? na 15-tym miejscu po przecinku.
		#tak na szybko, brute force obej?cie: sortuj?
		granice<-sort(as.vector(unique(Hmisc::wtd.quantile(x, prob=0:n/n, type='quantile', weights=weights))));
	}
	#oznaczam, do ktorego przedzialu nalezy dana obserwacja
	przedzial<-findInterval(x, granice, rightmost.closed = TRUE, all.inside = TRUE);
	
	#i licze potrzebne rzeczy
	od<-granice[1:(length(granice)-1)];
	do<-granice[2:length(granice)];		
	
	srodek<-as.vector(tapply(x, przedzial, median));
	
	n_obs<- as.vector(tapply(weights, przedzial, sum));
	mean<-as.vector(tapply(x*weights, przedzial, FUN=sum))/n_obs;
	n_default<- as.vector(tapply(y*weights, przedzial, sum));
	br<- n_default/n_obs;
	logit<-log(br/(1-br));
	probit<-qnorm(br);
	
	#z tego co pami?tam, mamy dzi?ki temu wyrzuci? puste przedzia?y
	zostaw<-sort(unique(przedzial));
	
	as.data.frame(cbind(nr=zostaw,  od=od[zostaw], do=do[zostaw], 
					srodek, mean,  n_default, n_obs, br, logit, probit, var=br*(1-br)/n_obs))
}

#' Dzieli na przedziały jedną z dwóch metod i wylicza na nich bad rate (dev). 
#'
#' Wersja w developmencie.
#' Może poza tym, że jeśli nie ma obserwacji w jakimś bukcecie, to go usuwa.
#' Aby ustrzec się przed problemami numerycznymi, wartości \code{x} zaokrąglane są do \code{digits}
#' miejsc po przecinku.
#'
#' @param x  Zmienna, którą będzie dyskretyzowana.  
#' @param y  Zmienna dwumianowa.  
#' @param n  Liczba wynikowych przedziałów.  
#' @param weights wektor z wagami.
#' @param method  Sposób podziału na przedziały. Szczegóły w sekcji Details.  
#' @param one.value.action Jeszcze nie działa.
#' @param avg \code{vector}, \code{data.frame}, \code{data.table} z wartościami dla których
#' zostaną wyliczone średnie.
#' @param total czy dołączyć wiersz z podsumowaniem.
#' @param sort_x jeśli \code{TRUE}, wynikowa tabela zostanie posortowana po wartościach 
  #' @param digits liczba miejsc po przecinku do których zostanie zaokrąglona wartość \code{x}.
#' @return  
#' Zwraca \code{data.frame}, w którym dla \code{i}-tego wiersza podane są statystyki
#' dla \code{i}-tego przedziału.
#' @author Michał Danaj
#' @export
bckt_br<-function(x, y, n, weights=rep(1,length(x)), 
                  method=c("eq_length", "eq_count"), one.value.action=c("none","combine"),
                  avg=NULL ,total = TRUE, sort_x=TRUE, digits=13)
{                                         
  #	TODO - domy?lna warto?? method
  method<-match.arg(method);
  
  #dolny kraniec bucketa (wartosc x)
  od=c(1:n);
  #gorny kranieb bucketa (wartosc x)
  do=c(1:n);
  
  x <- round(x, digits=digits)
  
  if (method=="eq_length"){
    granice<-seq(min(x),max(x),length.out=n+1)
  }else{
    #w przypadku, gdy jedna warto?? jest dla wielu kwantyli, zdarzaj? si? problemy numeryczne
    #?e warto?? teoretycznie jest taka sama, ale r??ni si? na 15-tym miejscu po przecinku.
    #tak na szybko, brute force obej?cie: sortuj?
    granice<-sort(as.vector(unique(Hmisc::wtd.quantile(x, prob=0:n/n, type='quantile', weights=weights))));
  }
  
  dt<-data.table(x,y,weights)
  buckety <- bckt_stat2(breaks=granice, dt=dt, avg=avg, total = total, sort_x=sort_x)
  return(buckety)
}


#' Wylicza statystyki zmiennej \code{default} dla podanych grup \code{buckets}
#'
#' Wylicza statystyki zmiennej \code{default} dla podanych grup \code{buckets} 
#' @usage buckety_stat(bucket, default, total=TRUE)
#  buckety_stat_wtd(bucket, default, weights=rep(1,length(bucket)), total=TRUE)
#' @param bucket wektor z identyfikatorami bucket?w.
#' @param default wektor zmiennych numerycznych.
#' @param total czy do??czy? wiersz z podsumowaniem.
#' 
#' @return
#'   Zwraca \code{data.frame}, w kt?rym dla \code{i}-tego wiersza podane s? statystyki
#'   dla \code{i}-tego przedzia?u.
#'   \item{nr }{Numer przedzia?u.}
#'   \item{od }{Pocz?tek przedzia?u.}
#'   \item{srodek}{?rodek przedzia?u.}
#'   \item{mean}{?rednia z obserwacji w przedziale.}
#'   \item{do}{Koniec przedzia?u.}
#'   \item{n_default}{Liczba default?w.}
#'   \item{n_obs}{Liczba obserwacji.}
#'   \item{br}{Bad rate.}
#'   \item{logit}{Logit.}
#'   \item{probit}{Probit.}
#'   \item{var}{Wariancja zmiennej \code{default}.}
#'   \item{waga}{Waga jako odwrotno?? warinacji. Gdy \code{n_default=0} to przyjmuje si?,
#' 			?e \code{n_default=0.5}}
#' @author Micha? Danaj 
#' @export
buckety_stat<-function (bucket, default, total = TRUE)
{
	
	if (any(is.na(bucket)) || any(is.na(default)))
		stop("W argumentach funkcji pojawi?y si? braki danych.")
	
	if (is.factor(bucket))
		bucket <- factor(bucket)
	
	obs <- table(bucket)
	obs_all <- sum(obs)
	bad <- as.vector(tapply(default, bucket, sum))
	bad_all <- sum(bad)
	br <- as.vector(tapply(default, bucket, mean))
	std_dev <- as.vector(tapply(default, bucket, sd))
	logit <- log(br/(1 - br))
	probit <- qnorm(br)
	waga <- NA
	pct_good = (obs - bad)/(obs_all - bad_all)
	pct_bad = bad/bad_all
	woe = log(pct_good/pct_bad)
	
	wynik <- as.data.frame(cbind(nr = 1:length(obs), n_good = (obs -
								bad), pct_good, n_bad = bad, pct_bad, n_obs = obs, pct_obs = obs/obs_all,
					gb_odds = (obs - bad)/bad, br, std_dev, woe, logit, probit,
					var = br * (1 - br)/obs, waga))
	
	rownames(wynik) <- sort(unique(bucket))
	if (total) {
		wynik <- rbind(wynik, TOTAL = c(length(obs) + 1, obs_all -
								bad_all, 1, bad_all, 1, obs_all, 1, (obs_all - bad_all)/bad_all,
						bad_all/obs_all, sd(default), 0, NA, NA, NA, NA))
	}
	wynik$label<-rownames(wynik);
	return(wynik)
}

#TODO za?o?y? ?e jak jest 0, o da? 0.5 do wyliczenia woe i logit
#' Wylicza statystyki zmiennej \code{y} dla podanych grup \code{x} (dev)
#'
#' Wylicza statystyki zmiennej \code{y} dla podanych grup \code{x}. W za?o?eniu
#' \code{y} jest zmienn? dwumianow? o warto?ciach {0,1}.
#' 
#' @param x wektor z identyfikatorami bucket?w.Nie mo?e by? brak?w danych.
#' @param y wektor zmiennych numerycznych. Z za?o?enia zmienna dwumianowa, ale statystyki
#' zostan? policzone dla ka?dej innej zmiennej numerycznej.Nie mo?e by? brak?w danych.
#' @param weights wektor z wagami.
#' @param dt \code{data.table} z kolumnami \code{x}, \code{y}, \code{weights}.
#' @param avg \code{vector}, \code{data.frame}, \code{data.table} z warto?ciami dla kt?rych
#' zostan? wyliczone ?rednie.
#' @param total czy do??czy? wiersz z podsumowaniem.
#' @param sort_x je?li \code{TRUE}, wynikowa tabela zostanie posortowana po warto?ciach 
#' cechy \code{x}. W przeciwnym wypadku sortowanie b?dzie zgodne z porz?dkiem w danych.
#' 
#' @return Zwraca \code{data.table} ze statystykami.
#' @author Micha? Danaj 
#' @export
bckt_stat<-function (x=NULL, y=NULL, weights=rep(1, length(x)), 
                     dt=NULL, avg=NULL ,total = TRUE, sort_x=TRUE)
{
  
  #składam dt
  if (!is.null(x))
    dt <- data.table(x, y, weights)

  #sprawdzam braki danych
  if (any(is.na(dt$x)) || any(is.na(dt$y)))
    stop("W argumentach funkcji pojawi?y si? braki danych.")
  
  #sprawdzam, czy są ujemne wartości wag
  if (any(dt$weights<0))
    warning("UWAGA! Ujemne wartości wag w funkcji bckt_stat.")
  
  #usuwam factora
  if (is.factor(dt$x)){
    dt$x <- factor(dt$x)
  }
  
  
  #jeśli zostały podane kolumny 'avg', liczymy od razu dla nich średnie
  if(!is.null(avg)){
    dt_avg <- data.table(x=dt$x, avg)
    #TODO ten lapply strasznie wolno działa. Popatrze, czy nie da się czegoś
    #z tym zrobi?
    dt_avg <- dt_avg[,lapply(.SD, mean), x][,-1]
  }
  
  #podsawowe statysyki dla wartości x
  dt_wyn <- dt[, .(
    n_good = sum(weights) - sum(y*weights)
    ,n_bad = sum(y*weights)
    ,n_obs = sum(weights)
  ), x]

  #podstawowe statystyki dla totala
  totals <- dt_wyn[,.(good_all=sum(n_good), bad_all=sum(n_bad), obs_all=sum(n_obs))]  
  
  #docelowa struktura tabeli wraz z wyliczeniami
  dt_wyn<- dt_wyn[,.(
    nr=NA
    ,label=as.character(x)
    ,discret=x
    ,od=NA
    ,srodek=NA
    ,do=NA
    ,mean=NA
    ,median=NA
    ,n_good
    ,pct_good = n_good/totals$good_all
    ,n_bad
    ,pct_bad = n_bad/totals$bad_all
    ,n_obs
    ,pct_obs = n_obs/totals$obs_all
    ,br = n_bad/n_obs
    ,logit = fifelse(n_obs==0 | n_bad==0, NaN, log(n_bad/n_obs))
    ,woe = log(n_bad/n_good)
    )]
  
  rownames(dt_wyn) <- dt_wyn$labels
  
  #sortowanie jeśli miało być
  if (sort_x==TRUE){
    setorder(dt_wyn, discret)
  }
  
  
  #wyliczam wiersz z Totalem
  if (total) {
    
    
    #Dla dodatkowych kolumn
    if (!is.null(avg)){
      totals_avg<-as.data.frame(lapply(dt_avg, function(x)sum(x*dt_wyn$n_obs)/sum(dt_wyn$n_obs)))
      dt_avg<-rbindlist(list(dt_avg, data.table(totals_avg)))
    }

    #zachowuję typ zmiennej. Muszę to uwzględnić w totalu
    if (typeof(x)=='character')
      discr_total=''
    else
      discr_total=NA
    
    #Dla zwyk?ych kolumn
    totals <- totals[,.(n_good=good_all, n_bad=bad_all, n_obs=obs_all)][,.(
      nr=NA
      ,label='TOTAL'
      ,discret=discr_total
      ,od=NA
      ,srodek=NA
      ,do=NA
      ,mean=NA
      ,median=NA      
      ,n_good
      ,pct_good = n_good/totals$good_all
      ,n_bad
      ,pct_bad = n_bad/totals$bad_all
      ,n_obs
      ,pct_obs = n_obs/totals$obs_all
      ,br = n_bad/n_obs
      ,logit = fifelse(n_obs==0 | n_bad==0, NaN, log(n_bad/n_obs))
      ,woe = log(n_bad/n_good)
    )]
    
    dt_wyn<-rbindlist(list(dt_wyn, totals))
    
  }
  
  dt_wyn$nr=seq(nrow(dt_wyn))
  
  #Jeśli były dodatkowe kolumy, to je tutaj dodaję
  if (!is.null(avg)){
    dt_wyn <- cbind(dt_wyn ,dt_avg)
  }
  
  #attr(dt_wyn,'typeof_x')=typeof(x)
  #attr(dt_wyn,'typeof_x')=class(x)
  
  
  return(dt_wyn)
}


#' Wylicza statystyki dla podanych przedzia??w
#'
#' Dzieli zmienn? \code{score} na przedzia?y okre?lone przez \code{breaks}
#' i wylicza statystyki dla tak powsta?ych bucket?w przy pomocy funkcji.
#' 
#' Do przypisania warto?ci do przedzia?u u?ywana jest funkcja
#' 
#' \code{findInterval(score, breaks, rightmost.closed = TRUE, all.inside = TRUE)}.
#' 
#' @param breaks punkty podzia?u
#' @param score zmienna score'owa.
#' @param def zmienna odpowiedzi z zakresu [0,1]. Np. default, LGD.
#' @param total czy wylicza? wiersz z podsumowaniem.
#' @seealso \code{\link{buckety_stat}}.
#' @return \code{data.frame} ze statystykami.
#' @author Micha? Danaj
#' @export
buckety_stat2<-function(breaks, score, def, total=FALSE){
	
	
	if (any(is.na(breaks)) || any(is.na(score)) || any(is.na(def)))
		stop("W argumentach funkcji pojawi?y si? braki danych.");
	
	zakres_danych<-range(na.omit(score))
	zakres_breaks<-range(breaks)
	if (zakres_danych[1]<zakres_breaks[1] || 
			zakres_danych[2]>zakres_breaks[2])
		warning("buckety_stat2: Zmienna 'score' spoza zakresu 'breaks'");
	
	breaks<-sort(unique(breaks));
	
	#podzial<-cut(score, breaks, include.lowest=TRUE);
	
	przedzial_nr<-findInterval(score, breaks, rightmost.closed = TRUE, all.inside = TRUE);
	
	nawiasy_koniec<-c(rep(")", length(breaks)-2),"]")
	
	od<-breaks[-length(breaks)]
	do<-breaks[-1]
	
	labels<-paste("[",od,", ", do, nawiasy_koniec,sep="")
	#Wybieram tylko te przedzia?y, w kt?rych s? jakie? dane. Mo?e by? z tym troch?
	#kicha. Zobaczymy...
	finalne_przedzialy<-sort(unique(przedzial_nr))
	labels<-labels[finalne_przedzialy]
	
	podzial<-factor(przedzial_nr, labels=labels, ordered = TRUE)
	
	buckety<-buckety_stat(podzial, def, total=total);
	buckety$od<-od[finalne_przedzialy];
	buckety$do<-do[finalne_przedzialy];
	
	buckety$srodek<-(buckety$od+buckety$do)/2;
	buckety$median<-tapply(score, podzial,median);
	buckety$mean<-tapply(score, podzial,mean);
	
	rownames(buckety)<-labels;
	buckety$label<-labels;

	return(buckety);
}


#' Wylicza statystyki dla podanych przedzia??w (dev)
#'
#' Dzieli zmienn? \code{x} na przedzia?y okre?lone przez \code{breaks}
#' i wylicza statystyki dla tak powsta?ych bucket?w.
#' 
#' Do przypisania warto?ci do przedzia?u u?ywana jest funkcja
#' 
#' \code{findInterval(x, y, rightmost.closed = TRUE, all.inside = TRUE)}.
#' 
#' @param breaks punkty podzia?u
#' @param x zmienna score'owa.
#' @param y zmienna odpowiedzi z zakresu [0,1]. Np. default, LGD.
#' @param weights wektor z wagami.
#' @param dt \code{data.table} z kolumnami \code{x}, \code{y}, \code{weights}.
#' @param avg \code{vector}, \code{data.frame}, \code{data.table} z warto?ciami dla kt?rych
#' zostan? wyliczone ?rednie.
#' @param total czy do??czy? wiersz z podsumowaniem.
#' @param sort_x je?li \code{TRUE}, wynikowa tabela zostanie posortowana po warto?ciach 
#' 
#' @seealso \code{\link{buckety_stat}}.
#' @return \code{data.frame} ze statystykami.
#' @author Micha? Danaj
#' @export
bckt_stat2<-function(breaks, x=NULL, y=NULL, weights=rep(1,length(x)), dt=NULL, avg=NULL,
                     total = TRUE, sort_x=TRUE){

  #Składam dt
  if (!is.null(x))
    dt <- data.table(x, y, weights)
  
  #Sprawdzam braki danych
  if (any(is.na(breaks)) || any(is.na(dt$x)) || any(is.na(dt$y)))
    stop("W argumentach funkcji pojawi?y si? braki danych.");
  
  #Sprawdzam czy x-y wychodzą poza zakres breaks
  zakres_danych<-range(na.omit(dt$x))
  zakres_breaks<-range(breaks)
  if (zakres_danych[1]<zakres_breaks[1] || 
      zakres_danych[2]>zakres_breaks[2])
    warning("bckt_stat2: Zmienna 'x' spoza zakresu 'breaks'");
  
  #porządkuję i definiuję przedziały
  breaks<-sort(unique(breaks));
  bck_od<-breaks[-length(breaks)]
  bck_do<-breaks[-1]

  #Znajduję i przypisuję w który przedział wpada wartość x-a
  dt$przedzial_nr<-findInterval(dt$x, breaks, rightmost.closed = TRUE, all.inside = TRUE)
  dt[,':='(x_orig=x, x=przedzial_nr)]
  
  #Wylicza staystyki na bucketach
  buckety_wyn<-bckt_stat(dt=dt, avg=avg, total = total, sort_x=sort_x);
  
  #doliczam medianę i średnią zmiennej
  temp<-dt[,.(median=median(x_orig), mean=mean(x_orig)), przedzial_nr]
  if (total)
    temp_tot<-dt[,.(median=median(x_orig), mean=mean(x_orig))]
  setorder(temp, przedzial_nr)

  #doliczam labele
  nawiasy_koniec<-c(rep(")", length(breaks)-2),"]")
  labels<-paste("[",bck_od,", ",bck_do, nawiasy_koniec,sep="")
  
  #niektóre buckety mogły być puste i zostały usunięte w bckt_stat. 
  #Muszę je usunąć z wektorów dołączanych do tabeli wynikowej
  labels <- labels[buckety_wyn$nr]
  bck_od=bck_od[buckety_wyn$nr]
  bck_do=bck_do[buckety_wyn$nr]

  #Poprzednia funkcja nie wiedziała, że mamy przedziały i jak wypełnić tymi przedziałami warości
  #Robię to tutaj
  if (total)
    buckety_wyn[,':='(
      label=c(labels,'TOTAL'),
      od=c(bck_od,NA),
      srodek=c((bck_od+bck_do)/2, NA),
      do=c(bck_do,NA),
      discret=NA,
      mean=c(temp$mean, temp_tot$mean),
      median=c(temp$median, temp_tot$median)
    )]
  else
    buckety_wyn[,':='(
      label=c(labels),
      od=c(bck_od),
      srodek=c((bck_od+bck_do)/2),
      do=c(bck_do),
      discret=NA,
      mean=c(temp$mean),
      median=c(temp$median)
    )]      
  
  return(buckety_wyn)
}


#' ??czy ze sob? buckety
#' 
#' ??czy buckety. W przypadku bucket?w z przedzia?ami zmiennej ci?g?ej, mo?liwe jest 
#' po??czenie tylko
#' przedzia??w przylegaj?cych do siebie. Dla nowo powsta?ych bucket?w wylicza statystyki. 
#' W przypadku ??czenia przedzia??w zmiennej ci?g?ej, wiersze z tymi bucketami zostan? ze sob?
#' po??czone i powstanie \code{data.frame} z liczb? wierszy o jeden mniejsz? ni? w \code{bucket}.
#' Przy ??czeniu bucket?w dyskretnych lub dyskretnego i ci?g?ego, wiersze nie zostan? usuni?te. 
#' Zostanie im nadany wsp?lny label oraz wsp?lny numer. Je?li liczba bucket?w do po??czenia jest<=1,
#' zwracany jest wej?ciowy podzia?. 
#' @param x zmienna score'owa.
#' @param y zmienna odpowiedzi z zakresu [0,1]. Np. default, LGD.
#' @param buckets buckety.
#' @param row_idxs numery wierszy w \code{buckets} do po??czenia.
#' @param new_label label jaki b?dzie nadany po??czonym bucketom. W przypadku braku podania, zostanie zatosowany domy?lny.
#' @return \code{data.frame}.
#' @author Micha? Danaj
#' @export
polacz_buckety<-function(x, y, buckets, row_idxs, new_label=NULL)
{
	row_idxs<-sort(unique(row_idxs));
	#sprawdzam, czy jest co ??czy?
	if (length(row_idxs)<=1)
		return(buckets)
	#sprawdzam, czy nie wyszli?my poza zakres
	if(min(row_idxs)<1 | max(row_idxs)>nrow(buckets)-1) 
		stop("Numery wierszy s? poza zakresem zmiennej 'buckets'");
	#sprawdzam, czy nie ma ju? takiego labela w innych wierszach
	if (!is.null(new_label) & any( buckets$label[-row_idxs]==new_label))
		warning("Podana warto?? 'new_label' znajduje si? ju? w 'buckets' w wierszu innym ni? aktualnie ??czone wiersze.")
	for (i in 1:(length(row_idxs)-1)){
		nr1<-row_idxs[i];
		nr2<-row_idxs[i+1];
		#je?li ??czymy dane ci?g?e
		if (all((!is.na(buckets$od) & !is.na(buckets$do))[c(nr1,nr1)])){
			if (nr2-nr1!=1)
				stop("B??dnie podane wiersze do po??czenia! Przedzia?y powinny by? do siebie przyleg?e!")
			#je?li label nie podany, to go tworz?
			if (is.null(new_label)){
				new_label<-strsplit(buckets$label[nr1],',')[[1]][1]
				new_label<-paste(new_label,strsplit(buckets$label[nr2],',')[[1]][2],sep=',')
			}
			#jeszcze przepisuj? kra?ce przedzia??w i inne warto?ci nie wyliczane w buckety_stat a
			#wyliczane w buckety_stat2
			buckets$do[nr1]<-buckets$do[nr2];
			buckets$srodek[nr1]<-c(buckets$od[nr1]+buckets$do[nr1])/2;
			buckets$label[nr1]<-new_label;
			rownames(buckets)[nr1]<-new_label;
			#usuwam drugi bucket
			buckets<-buckets[-nr2,];
			
		}else{
			if (is.null(new_label))
				new_label<-paste(buckets$label[nr1],buckets$label[nr2],sep=',');
			buckets$label[c(nr1,nr2)]<-new_label;
			buckets$nr[nr2]<-buckets$nr[nr1];
		}
		buckets$fitted<-buckets$label;
		x_buckets<-przypisz2(x,buckets);
		buckets_new<-buckety_stat(x_buckets, y)
		buckets<-cbind(buckets[,c('nr','label','discret','od','srodek','do')], 
				buckets_new[buckets$label,c('n_good','pct_good','n_bad','pct_bad','n_obs','pct_obs','br','woe','logit')]);
	}	
	return(buckets);
}
