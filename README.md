# UDP - Student Mobility
## 1. Import and clean raw data, combine to create analytical files
<dl>
  <dt>analytical_files/tps_covariates.py </dt>
  <dd>Import 'student_geos_<year>.csv' (base) files (provided by TPS)</dd>
  <dd>Aggregate files together, reshape (long by tract, wide by year)</dd>
  <dd>Export 'tps_student_counts_by_tract.csv'</dd>
  <br />
  <dt>analytical_files/tulsa_parcels.py </dt>
  <dd>Import 'TulsaParcelsNeighborhood.csv' (base) file (provided by CoT)</dd>
  <dd>Keep desired fields</dd>
  <dd>Export 'tulsa_parcels.csv'</dd>
  <br />
  <dt>analytical_files/tulsa_parcel_tps_tract_association.Rmd </dt>
  <dd>Import 'tulsa_parcels.csv'</dd>
  <dd>Associate parcel geocodes to geotracts (same as that in 'tps_student_counts_by_tract.csv'</dd>
  <dd>Export 'tulsa_parcel_tps_tract_association.csv'</dd>
  <hr />
  <dt>analytical_files/base_analytical_file.py</dt>
  <dd>Import 'tps_student_counts_by_tract.csv' (Student Count Data)</dd>
  <dd>Import 'tulsa_parcel_tps_tract_association.csv' (Parcel Data)</dd>
  <dd>Filter, normalize, reshape, split (by parcel types)</dd>
  <dd>Merge together parcel and student count data</dd>
  <dd>Export 'base_file_&lsaquo;wide/long&rsaquo;_&lsaquo;residential/commercial/combo&rsaquo;.csv' (analytical) files</dd>
</dl>

## 2. Run Analysis on analytical files
<dl>
  <dt>analysis/feature_selection.py</dt>
  <dt>analysis/udp_feature_selection.py</dt>
</dl>
  
