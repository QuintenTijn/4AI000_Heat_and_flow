(* qmmlpack                      *)
(* (c) Matthias Rupp, 2006-2016. *)
(* See LICENSE.txt for license.  *)

description = "Linear kernel vector m, Mathematica/C++/BLAS";

ngrid = {10, 50, 100, 500, 1000, 5000, 10000};  (* Number of training data *)
dgrid = {1, 10, 50, 100, 300, 500, 1000, 10000};  (* Dimensionality of training data *)
theta = {{"n", ngrid}, {"d", dgrid}};  (* thetaGrid computed automatically *)

(* Creating input samples for all theta values via                  *)
(* xx = Map[RandomReal[{-100, 100}, #] &, theta, {-2}];             *)
(* occupies > 1.5GB RAM, leading to swapping and distorting results *)

before[thetaval_, thetaind_] := (
    xx = RandomReal[{-100, 100}, thetaval];
    If[Not[Developer`PackedArrayQ[xx]], bmError[StringFrom["error creating input data for theta = ``.", thetaind]]];
);

after[thetaval_, thetaind_, result_] := (
    If[Not[VectorQ[result, NumberQ]], bmError[StringForm["non-vector result for theta = ``.", thetaind]]];
    xx = None; ClearSystemCache[];
);

function[thetaval_, thetaind_] := KernelLinear[xx, {}, Diagonal];
