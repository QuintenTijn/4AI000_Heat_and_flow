(* qmmlpack                      *)
(* (c) Matthias Rupp, 2006-2016. *)
(* See LICENSE.txt for license.  *)

description = "Linear kernel matrix L, Mathematica/C++/BLAS";

ngrid = {10, 50, 100, 500, 1000, 5000, 10000};  (* Number of training data *)
dgrid = {1, 10, 50, 100, 300, 500, 1000, 10000};  (* Dimensionality of training data *)
theta = {{"n", ngrid}, {"d", dgrid}};  (* thetaGrid computed automatically *)

(* Creating input samples for all theta values via                  *)
(* xx = Map[RandomReal[{-100, 100}, #] &, theta, {-2}];             *)
(* occupies > 1.5GB RAM, leading to swapping and distorting results *)

before[thetaval_, thetaind_] := (
    xx = RandomReal[{-100, 100}, thetaval];
    zz = RandomReal[{-100, 100}, thetaval];
    If[Not[Developer`PackedArrayQ[xx]], bmError[StringFrom["error creating X input data for theta = ``.", thetaind]]];
    If[Not[Developer`PackedArrayQ[zz]], bmError[StringFrom["error creating Z input data for theta = ``.", thetaind]]];
);

after[thetaval_, thetaind_, result_] := (
    If[Not[MatrixQ[result, NumberQ]], bmError[StringForm["non-matrix result for theta = ``.", thetaind]]];
    If[Not[Dimensions[result] == {thetaval[[1]], thetaval[[1]]}], bmError["resultant L matrix has invalid dimensions"]];
    xx = None; zz = None; ClearSystemCache[];
);

function[thetaval_, thetaind_] := KernelLinear[xx, zz, {}];
