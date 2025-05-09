#!/usr/bin/env MathematicaScript -script

(* qmmlpack                      *)
(* (c) Matthias Rupp, 2006-2016. *)
(* See LICENSE.txt for license.  *)

(*  ***********************  *)
(*  *  Utility functions  *  *)
(*  ***********************  *)

(* Prints error message and aborts with non-zero exit code. *)
bmError[msg_] := (Print["Error: ", ToString[msg]]; Exit[1];)

(* Formats a time difference in seconds *)
bmFormatTime[time_] := Module[{d,h,m,s},
    t = Round[time]; (* Drop sub-second part *)
    d = Quotient[t, 86400]; t -= d*86400;  (* days, 60*60*24s *)
    h = Quotient[t,  3600]; t -= h* 3600;  (* hours, 60*60s *)
    m = Quotient[t,    60]; t -= m*   60;  (* minutes, 60s *)
    s = t;
    StringJoin[IntegerString[d], "d ", IntegerString[h,10,2], ":", IntegerString[m,10,2], ":", IntegerString[s,10,2]]
];

(* Prints progress indicator *)

bmShowProgress[theta_, ind_] := Module[{now, pos, eta, thetaflat, percent, percentMeter, output, updateFrequency = 3, meterSize = 20},
    now = AbsoluteTime[];
    If[Not[ValueQ[bmProgressStart]], bmProgressStart = now];
    If[Not[ValueQ[bmProgressLast ]], bmProgressLast  =  0 ];
    If[now - bmProgressLast < updateFrequency, Return[]];
    bmProgressLast = now;

    output = bmFormatTime[now - bmProgressStart];

    thetaflat = Flatten[theta, ArrayDepth[theta]-2];
    pos = Position[thetaflat, Extract[theta, ind], {1}, Heads->False][[1,1]];
    eta = N[(Length[thetaflat] - pos) * Length[thetaflat] / pos];
    output = output <> "  ETA " <> bmFormatTime[eta];

    percent = Round[100 * pos / Length[thetaflat]];
    percentMeter = Round[percent * meterSize / 100];
    output = output <> "  "
        <> StringPadRight["[", 1+percentMeter, "*"]
        <> StringPadLeft["]", 1+meterSize-percentMeter, "-"]
        <> " " <> IntegerString[percent] <> "%"
        <> " " <> ToString[thetaflat[[pos]]];

    Print[output];
];

(*  ********************************  *)
(*  *  Parse command line options  *  *)
(*  ********************************  *)

(* Help text *)
bmHelpText = "usage: " <> First[$ScriptCommandLine] <> " [-h] [-f] [-p] [-n] snippet" \
    <> "\n\nRuns timings for qmmlpack development.\n\n" \
    <> "positional arguments:\n" \
    <> "  snippet         benchmark code snippet to be timed\n\n" \
    <> "optional arguments:\n" \
    <> "  -h, --help      show this help message and exit\n" \
    <> "  -f, --force     overwrites existing results file\n" \
    <> "  -p, --progress  shows progress bar\n" \
    <> "  -n, --dryrun    does not write results file\n\n" \
    <> "Part of qmmlpack library.";

(* Default values of options *)
bmForce    = False;
bmProgress = False;
bmDryRun   = False;

(* Parse command line options *)
Module[{args, arg},
    args = Rest[$ScriptCommandLine];
    If[Length[args] == 0, Print[bmHelpText]; Exit[0]];
    While[Length[args] > 0,
        arg = First[args]; args = Rest[args];
        Switch[arg,
            "-h" | "--help"    , (Print[bmHelpText]; Exit[0]),
            "-f" | "--force"   , bmForce    = True,
            "-p" | "--progress", bmProgress = True,
            "-n" | "--dryrun"  , bmDryRun   = True,
            _?(StringQ[#] &),
                (If[ValueQ[bmSnippet], bmError[StringForm["second positional argument '``'.", arg]]]; bmSnippet = arg),
            _, bmError[StringForm["can not interpret command line argument '``'.", arg]]
        ];
    ];
];

(* Check command line arguments *)
If[Not[ValueQ[bmSnippet]], bmError["snippet not specified."]];
If[Not[FileExistsQ[bmSnippet]], bmError[StringForm["snippet file '``' not found.", bmSnippet]]];

(*  **********************  *)
(*  *  Set up variables  *  *)
(*  **********************  *)

(* Date and time the benchmark is run *)
bmDateTimeString = DateString[{"Year", "-", "Month", "-", "Day", "_", "Hour", "-", "Minute"}]

(* Machine on which benchmark is run *)
bmSystemDescr = StringJoin[Riffle[{$MachineName, $SystemID, "Mathematica " <> ToString[$VersionNumber]}, ", "]];

(* Filename with actual benchmark code *)
bmSnippet = ExpandFileName[bmSnippet];

(* Directory where benchmarking driver is located. Assumed qmmlpack library subfolder *)
bmBaseDir = ExpandFileName[FileNameDrop[First[$ScriptCommandLine], -1]];

(* Directory where result files are stored *)
bmResultsDir = FileNameJoin[{bmBaseDir, "results"}];
If[Not[DirectoryQ[bmResultsDir]], bmError[StringForm["could not access results directory '``'.", bmResultsDir]]];

(* File in which results are stored *)
bmResultsFile = FileBaseName[bmSnippet] <> "_" <> bmDateTimeString <> ".json";
bmResultsFile = FileNameJoin[{bmResultsDir, bmResultsFile}];
If[FileExistsQ[bmResultsFile] && Not[bmForce], bmError[StringForm["results file '``' already exists.", bmResultsFile]]];

(*  **********************  *)
(*  *  Benchmarked code  *  *)
(*  **********************  *)

(* Provide QMMLPack package *)
$Path = Append[DeleteCases[$Path, _?(StringContainsQ[#, "qmmlpack"] &)], FileNameJoin[{bmBaseDir, "..", "mathematica"}]];
Needs["QMMLPack`"]

(* Provide convenience routines *)
bmOuter[a_, b_] := Outer[List, a, b];

(* Execute benchmarked code snippets *)
Get[bmSnippet];

(* Process snippet definitions *)
If[Not[NameQ["function"]], bmError["snippet did not define function to be benchmarked."]];

If[Not[ValueQ[theta]], theta = None];

If[Not[ValueQ[thetaGrid]],
    thetaGrid = Which[
        theta === None, None,
        Length[theta] == 1, Transpose[{theta[[1,2]]}],
        Length[theta] == 2, bmOuter[theta[[1,2]], theta[[2,2]]],
        True, bmError["only up to two parameters are currently supported."]
    ];
    ,
    If[Not[NumberQ[TensorRank[thetaGrid]] || thetaGrid === None], bmError["snippet defined non-tensor theta."]]
];

If[Not[ValueQ[description]], bmError["snippet did not define description of benchmark."]];
If[Not[StringQ[description]], bmError["snippet defined non-String description of benchmark."]];

If[Not[NameQ["before"]], before[args___] := Null];
If[Not[NameQ["after" ]], after [args___] := Null];

(*  *****************  *)
(*  *  Run timings  *  *)
(*  *****************  *)

bmTimingFunction[] := Module[{time, result},
    ClearSystemCache[];
    before[];
    {time, result} = RepeatedTiming[function[], 3];
    after[result];
    time
];

bmTimingFunction[thetaval_, thetaind_] := Module[{time, result},
    ClearSystemCache[];

    before[thetaval, thetaind];
    {time, result} = RepeatedTiming[function[thetaval, thetaind], 3];
    after[thetaval, thetaind, result];
    If[bmProgress, bmShowProgress[thetaGrid, thetaind]];

    time
];

If[theta === None,
    bmTimings = bmTimingFunction[]
    ,
    bmTimings = MapIndexed[bmTimingFunction, thetaGrid, {-2}];
];

(*  ***************************  *)
(*  *  Export timing results  *  *)
(*  ***************************  *)

bmResults = {
    "DateTime"  -> bmDateTimeString,
    "Benchmark" -> description,
    "System"    -> bmSystemDescr,
    "Theta"     -> theta,
    "ThetaGrid" -> thetaGrid,
    "Timings"   -> bmTimings
};

If[Not[bmDryRun], Export[bmResultsFile, bmResults, "JSON", "Compact" -> False]];
