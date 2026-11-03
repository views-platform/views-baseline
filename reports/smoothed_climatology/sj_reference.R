# Reference values for tests/test_smoothed_climatology.py::test_sj_matches_r_bw_sj.
# Run: Rscript reports/smoothed_climatology/sj_reference.R   (R 4.6.2 used, 2026-11-03)
# nb and tol are raised so R's binning and root tolerance don't blur the reference;
# views-baseline's sheather_jones_bandwidth uses exact pairwise sums.
sets <- list(c(1,2,40), c(1,1,2,3,40), c(1,1,1,2), c(1,2), c(3,5,8,13,21,34),
             c(1,1,1,2,2,3,4,7,12,30,150), c(2,4,4,5,9,11,11,13,20,26,33,60,61,95),
             c(1,3,6,10,15,21,28,36,45,55), c(-1.2,0.3,0.8,-0.4,2.1,-0.9,0.05,1.4,-1.7,0.6))
for (x in sets) cat(paste(x, collapse = ","), sprintf("%.10f", bw.SJ(x, nb = 200000L, tol = 1e-14)), "\n")
