# Note: this requires pdoc>=14.1.0 to run (pip install pdoc>=14.1.0)
rm -rvf index.html outrank outrank.html search.js;
cd ..;
pdoc ./outrank -o docs;
