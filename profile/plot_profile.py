import numpy as np
import matplotlib.pyplot as plt

profile = np.fromfile('profile_nearest.bin', dtype=np.float32).reshape(4, -1)
profile_bilinear = np.fromfile('profile_bilinear.bin',
                               dtype=np.float32).reshape(4, -1)
sizes = 10 * np.array([1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6])

plt.plot(sizes, profile[0], 'r')
plt.plot(sizes, profile_bilinear[0], 'g')
plt.plot(sizes, profile[2], 'b')
plt.plot(sizes, profile[1], 'r--')
plt.plot(sizes, profile_bilinear[1], 'g--')
plt.plot(sizes, profile[3], 'b--')
plt.plot(sizes, profile[0] + profile[1], 'r:')
plt.plot(sizes, profile_bilinear[0] + profile_bilinear[1], 'g:')
plt.plot(sizes, profile[2] + profile[3], 'b:')
plt.legend([
    'Mapped (Nearest) Forward', 'Mapped (Bilinear) Forward', 'Grid Forward',
    'Mapped (Nearest) Backward', 'Mapped (Bilinear) Backward', 'Grid Backward',
    'Mapped (Nearest) Combined', 'Mapped (Bilinear) Combined', 'Grid Combined'
])
plt.xlabel('Input Size (# Elements)')
plt.ylabel('Time (s)')
plt.title('Mapped vs. Grid Convolution Running Time')
plt.show()
