import csv
import statistics

filename_k = "k_compression_rate.csv"
filename_v = "v_compression_rate.csv"

compression_rates_k = []
compression_rates_v = []

with open(filename_k, newline="") as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        compression_rate = float(row[" compression_rate"])
        compression_rates_k.append(compression_rate)

with open(filename_v, newline="") as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        compression_rate = float(row[" compression_rate"])
        compression_rates_v.append(compression_rate)

max_compression_rate_k = max(compression_rates_k)
min_compression_rate_k = min(compression_rates_k)
avg_compression_rate_k = statistics.mean(compression_rates_k)
max_compression_rate_v = max(compression_rates_v)
min_compression_rate_v = min(compression_rates_v)
avg_compression_rate_v = statistics.mean(compression_rates_v)

print(f"max key compression_rate: {max_compression_rate_k}")
print(f"min key compression_rate: {min_compression_rate_k}")
print(f"avg key compression_rate: {avg_compression_rate_k}")
print(f"max value compression_rate: {max_compression_rate_v}")
print(f"min value compression_rate: {min_compression_rate_v}")
print(f"avg value compression_rate: {avg_compression_rate_v}")
