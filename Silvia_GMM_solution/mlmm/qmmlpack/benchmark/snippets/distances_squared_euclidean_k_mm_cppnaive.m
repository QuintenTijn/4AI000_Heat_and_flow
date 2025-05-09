(* qmmlpack                      *)
(* (c) Matthias Rupp, 2006-2016. *)
(* See LICENSE.txt for license.  *)

description = "Squared Euclidean distance matrix K, Mathematica/C++/naive";  (* For loop implemenation *)

ngrid = {10, 50, 100, 500, 1000, 5000};  (* Number of training data *)
dgrid = {1, 10, 50, 100, 300, 500, 1000};  (* Dimensionality of training data *)
theta = {{"n", ngrid}, {"d", dgrid}};  (* thetaGrid computed automatically *)

before[thetaval_, thetaind_] := (
    xx = RandomReal[{-100, 100}, thetaval];
    If[Not[Developer`PackedArrayQ[xx]], bmError[StringFrom["error creating input data for theta = ``.", thetaind]]];
);

after[thetaval_, thetaind_, result_] := (
    If[Not[MatrixQ[result, NumberQ]], bmError[StringForm["non-matrix result for theta = ``.", thetaind]]];
    xx = None; ClearSystemCache[];
);

function[thetaval_, thetaind_] := DistanceSquaredEuclidean[xx];
