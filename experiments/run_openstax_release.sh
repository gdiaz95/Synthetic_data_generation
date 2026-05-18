echo "Generating OpenStax release data..."

python3 script/generate_openstax_release.py

echo "Generating comparison plots..."

python3 script/Plot_reports.py

echo "Done. Release CSVs and plot written to Openstax_test_data/"
