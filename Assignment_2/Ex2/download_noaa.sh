mkdir -p ./NOAA-065900
for year in $(seq 1949 2025); do
  wget -q "https://www.ncei.noaa.gov/pub/data/noaa/$year/065900-99999-$year.gz" \
       -P ./NOAA-065900/
done
