cap cd "D:\OneDrive\Documents\Macros"
cap cd "/mnt/d/OneDrive/Documents/Macros"
set more off
set rmsg off
set trace off
set linesize 200
cap set processors 8

timer clear
timer on 1

cap log close
qui log using "cmp\unit tests.log", replace

version 13

set seed 0193284710

cmp setup

use laborsup, clear

cmp (kids = fem_inc male_educ), ind($cmp_cont) quietly

cmp (kids = fem_inc male_educ) (fem_work = male_educ), ind($cmp_cont $cmp_cont) quietly

cmp (fem_educ = kids other_inc fem_inc) (male_educ = kids other_inc fem_inc), ind(1 1) qui

cmp (kids = fem_inc male_educ) (fem_work = kids fem_inc), ind($cmp_cont $cmp_cont) qui

cmp (kids = fem_inc male_educ other_inc) (fem_work = kids), ind($cmp_cont $cmp_cont) qui

cmp (kids = fem_inc male_educ), ind($cmp_probit) qui
predict p2, pr
margins, dydx(*) predict(pr)

cmp (kids = fem_inc male_educ), ind($cmp_oprobit) qui
margins, dydx(*) predict(eq(#1) outcome(#2) pr)

gen byte anykids = kids > 0
cmp (anykids = fem_inc male_educ) (fem_work = male_educ), ind($cmp_probit $cmp_probit)
preserve
replace anykids=1
replace fem_work=1
margins, dydx(fem_inc) expression(exp(predict(lnl))) force // marginal effect on probability of (1,1)
restore

cmp (anykids = ) (fem_work = ), ind($cmp_probit $cmp_probit) nolr qui

cmp (fem_work = other_inc fem_educ kids) (other_inc = fem_educ kids male_educ), ind($cmp_probit $cmp_cont)
margins, predict(pr) dydx(*) force

cmp (other_inc = fem_educ kids fem_work) (fem_work = male_educ), ind($cmp_cont $cmp_probit) qui

cmp (fem_inc = kids male_educ), ind("cond(fem_inc>10, $cmp_cont, $cmp_left)") qui
margins, dydx(*) predict(pr(17 .))

cmp (fem_inc=kids male_educ) (male_educ=kids other_inc fem_work), ind("cond(fem_inc>10,$cmp_cont,$cmp_left)" $cmp_cont)

use intregxmpl, clear
cmp (wage1 wage2 = age age2 nev_mar rural school tenure), ind($cmp_int) qui

use laborsub, clear
cmp (whrs = kl6 k618 wa we, trunc(0 .)), ind($cmp_cont) qui

use 401k, clear
cmp (prate = mrate ltotemp age i.sole), ind($cmp_frac) qui
margins, dydx(mrate) predict(pr)

use sysdsn3, clear
cmp (insure = age male nonwhite site2 site3, iia), nolr ind($cmp_mprobit) qui
margins, dydx(nonwhite) predict(eq(#2) pr)


use fitness, clear
gen byte hours_pos = hours > 0
cmp (hours = age i.smoke distance i.single, trunc(0 .)) (hours_pos = commute whours age), nolr ind("cond(hours_pos, $cmp_cont, $cmp_out)" $cmp_probit) covar(indep) qui

use travel, clear
drop invehiclecost traveltime partysize
reshape wide choice termtime travelcost, i(id) j(mode)
constraint 1 [air]termtime1 = [train]termtime2
constraint 2 [train]termtime2 = [bus]termtime3
constraint 3 [bus]termtime3 = [car]termtime4
constraint 4 [air]travelcost1 = [train]travelcost2
constraint 5 [train]travelcost2 = [bus]travelcost3
constraint 6 [bus]travelcost3 = [car]travelcost4
cmp (air:choice1=t*1) (train: choice2=income t*2) (bus: choice3=income t*3) (car: choice4=income t*4), ind((6 6 6 6)) constr(1/6) nodrop struct tech(dfp) ghkd(200)
predict cmppr1, eq(air) pr
predict cmppr2, eq(train) pr
predict cmppr3, eq(bus) pr
predict cmppr4, eq(car) pr

use wlsrank, clear
reshape wide rank high low, i(id) j(jobchar)
constraint 1 [esteem]high1=[variety]high2
constraint 2 [esteem]high1=[autonomy]high3
constraint 3 [esteem]high1=[security]high4
constraint 4 [esteem]low1=[variety]low2
constraint 5 [esteem]low1=[autonomy]low3
constraint 6 [esteem]low1=[security]low4
cmp (esteem:rank1=high1 low1)(variety:rank2=female score high2 low2)(autonomy:rank3=female score high3 low3)(security:rank4=female score high4 low4), ind((9 9 9 9)) tech(dfp) ghkd(47, type(hammersley) scramble) rev constr(1/6)
predict cmppr1, eq(esteem) pr
predict cmppr2, eq(variety) pr
predict cmppr3, eq(autonomy) pr
predict cmppr4, eq(security) pr

use class10, clear
cmp (graduate = program#(c.income roommate c.hsgpa) program income roommate hsgpa) (program = i.campus i.scholar income) (hsgpa = income i.hscomp), vce(robust) ind(4 4 1) qui nolr

use womenwk, clear
gen selectvar = wage<.
cmp (wage = education age) (selectvar = married children education age), ind(selectvar $cmp_probit) nolr qui
margins, dydx(*) predict(e eq(wage) condition(0 ., eq(selectvar)))
margins, dydx(*) expression(predict(e eq(wage) cond(0 ., eq(selectvar))) * predict(pr eq(selectvar)) )
predict xb, eq(selectvar) xb
predict e, eq(selectvar) e(0 .)
gen cmp_mills = e - xb

gen wage2 = wage > 20 if wage < .
cmp (wage2 = education age) (selectvar = married children education age), ind(selectvar*$cmp_probit $cmp_probit) qui
margins, dydx(*) predict(pr eq(wage2) condition(0 ., eq(selectvar)))

gen wage3 = (wage > 10)+(wage > 30) if wage < .
cmp (wage3 = education age) (selectvar = married children education age), ind(selectvar*$cmp_oprobit $cmp_probit) nolr qui

sysuse auto, clear

cmp (price = foreign#) (foreign = mpg), ind($cmp_cont $cmp_cont) nolr
predict pricehat1
predict pricehat2, reducedform

cmp (price = foreign#) (foreign = mpg), ind($cmp_cont $cmp_probit)

replace foreign = . in 1/20
cmp (price = foreign#) (foreign = mpg), ind($cmp_cont $cmp_probit) // sample does not shrink

use klein, clear
cmp (consump = wagepriv# wagegovt) (wagepriv = consump# govt capital1), ind($cmp_cont $cmp_cont) nolr tech(dfp) qui

use supDem, clear
cmp (price = quantity# pcompete income) (quantity = price# praw), ind($cmp_cont $cmp_cont) nolr tech(dfp) qui
cmp, resultsform(reduced)
margins, dydx(*) predict(eq(quantity))

egen priceO = cut(price), at(25 27 31 33 35 37 39) icodes
egen quantityO = cut(quantity), at(5 7 9 11 13 15 17 19 21) icodes
cmp (price: priceO = quantity# pcompete income) (quantity: quantityO = price# praw), ind($cmp_oprobit  $cmp_oprobit) nolr qui tech(dfp)
cmp, resultsform(reduced)
margins, dydx(praw) predict(outcome(3) eq(quantity) pr)


use union, clear
cmp (union = age grade not_smsa south##c.year || idcode:), ind($cmp_probit) qui

use nlswork3, clear
replace ln_wage = 1.9 if ln_wage > 1.9
cmp (ln_wage = union age grade not_smsa south south#c.year || idcode:), ind("cond(ln_wage<1.899999, $cmp_cont, $cmp_right)") nolr qui

use nlswork5, clear
cmp (ln_wage1 ln_wage2 = union age grade south south#c.year occ_code || idcode:), ind($cmp_int) nolr qui

use tvsfpors, clear
cmp (thk = prethk cc#tv || school: || class:), ind($cmp_oprobit) intpoints(7 7) nolr qui

use productivity, clear
cmp (gsp = private emp hwy water other unemp || region: || state:), nolr ind($cmp_cont)

use womenhlthre, clear
gen byte goodhlth = health > 3
cmp (goodhlth = insured#c.(exercise grade) exercise grade insured || personid:) (insured = grade i.workschool || personid:), ind($cmp_probit $cmp_probit) intp(7) nolr

use wagework, clear
cmp (wage = age tenure || personid:) (working = age market || personid:), ind(working*$cmp_cont $cmp_probit) intp(7) nolr


use laborsup, clear

gen byte kids2 = kids + int(uniform()*3)
gen byte kids3 = kids + int(uniform()*3)
cmp (kids=fem_educ) (kids2=fem_educ) (kids3=fem_educ), ind($cmp_oprobit $cmp_oprobit $cmp_oprobit) nolr qui

cmp (other_inc = fem_work) (fem_work = kids), ind($cmp_cont $cmp_probit) qui robust

gen byte ind2 = cond(fem_work, cond(fem_inc, $cmp_cont, $cmp_left), $cmp_out)
cmp (other_inc=fem_inc kids) (fem_inc=fem_educ), ind($cmp_cont ind2)

use gcse, clear
cmp (gcse = lrt || school: lrt), ind($cmp_cont) nolr

use jspmix, clear
cmp (tby = sex, iia || scy3:), ind($cmp_mprobit) nolr

use union, clear
cmp (union = age not_smsa black# || idcode:) (black = south#c.year), ind($cmp_probit $cmp_probit) nolr


use laborsup, clear

gen byte kids2 = kids + int(uniform()*3)
cmp (kids=fem_educ) (kids2=fem_educ), ind($cmp_oprobit $cmp_oprobit) nolr tech(dfp) qui

predict xbA

predict xbB*
predict xbC xbD

predict sc*, score

predict lnl, lnl

predict prA, pr outcome(0)
predict prB, outcome(#1)

predict prC, outcome(4) eq(kids2)

predict prD*, pr

predict prE prF, pr

predict prG, eq(#2) pr

qui log close

timer off 1
timer list
