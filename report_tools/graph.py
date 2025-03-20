import matplotlib.pyplot as plt

datasets = ['Adult', 'Compas', 'Law School', 'Kdd', 'Dutch', 'Credit', 'Crime', 'German']
rs_idi_avg = [0.423, 0.065, 0.000, 0.033, 0.020, 0.306, 0.000, 0.082]
ga_idi_avg = [0.891, 0.795, 0.054, 0.457, 0.728, 0.796, 0.000, 0.839]

plt.figure(figsize=(10, 5))
plt.plot(datasets, rs_idi_avg, marker='o', linestyle='-', label='Random Search (RS)')
plt.plot(datasets, ga_idi_avg, marker='s', linestyle='-', label='Genetic Algorithm (GA)')

plt.title('Average IDI Ratio: Genetic Algorithm vs Random Search')
plt.xlabel('Dataset')
plt.ylabel('Average IDI Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('idi.pdf', dpi=300)
plt.show()
