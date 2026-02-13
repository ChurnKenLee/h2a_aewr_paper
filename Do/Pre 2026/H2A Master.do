** H2A Project: Master File
** Philip Hoxie
** 11/21/23

/*

This file organizes the sub-master do files so the project can be easily re-run. 

*/

clear 
matrix drop _all
scalar drop _all
estimates drop _all

** Macros **

global Do "R:/Hoxie/H-2A Paper/Do"
global Data "R:/Hoxie/H-2A Paper/Data Int"
global Output "R:/Hoxie/H-2A Paper/Output"

** Load and Clean Data **

cd "$Do"
do "H2A Clean and Load.do" // Cleans and Loads data sets from unpacked binaries folder

** Merge **

cd "$Do"
do "H2A Merge.do" // Merges datasets together, makes major variables

** Pre-analysis cleaning **

** Analysis Files (no cleaning or new variables here) **