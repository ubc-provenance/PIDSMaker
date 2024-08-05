# Dadabase Creating

There are 12 DARPA datasets in total, one of which is empty:
- E3: `CADETS`, `THEIA`, `TRACE`, `CLEARSCOPE` and `FIVEDIRECTIONS`
- E5: `CADETS`, `THEIA`, `TRACE`, `CLEARSCOPE`, `FIVEDIRECTIONS`, `MARPLE`
and `STARC (empty)`

So we can have up to 11 datasets available.

## Update 07-22
As of July 22, I finished creating database for 8 of them: E3 and E5 pairs
of `CADETS`, `THEIA`, `TRACE` and `CLEARSCOPE`.

Among them:
1. Both `CADETS_E3` and `CADETS_E5` don't have `path` for all subjects 
and most files. And there is no detailed parameters of `cmdLine`.
2. `TRACE_E5` does not have complete `cmdLine` for subjects.
3. `CLEARSCOPE_E3` and `CLEARSCOPE_E5` dont have `path` for subjects. 
And `address` and `port` are not complete, especially remote address and 
port.
4. `THEIA_E3`, `THEIA_E5` and `TRACE_E3` have complete data fields.

In addition, file `path` of `FIVEDIRECTIONS_E3` and `FIVEDIRECTIONS_E5`
are in `Event` logs instead of  `FileObject` logs. So I failed to create
databases for them using old scripts. I modified codes and it is rerunning
now.

I finished generating `darpa_v4` ground truth for all 8 available datasets
and merged it into branch `dev`.