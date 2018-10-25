import numpy as np

# testbytes = np.fromstring('''80 01''', dtype=np.ubyte, count=2)

# testbytes = np.fromstring('''64 36 de 9c 7d c6 a0 76''', dtype=np.ubyte, count=8)

# testbytes = np.fromstring('''64 36 de 9c 7d c6 a0 76 07 8b 0d 78 2f 47 07 6c
# c0 bf 25 ed 9f 9a 9e e8 6e ff 5f b5 07 b6 57 05
# da 00 9a 62 ff 2f 0f e0 91 35 a4 cf 35 23 19 f2
# ''', dtype=np.ubyte, count=48)

# testbytes = np.fromstring('''64 36 de 9c 7d c6 a0 76 07 8b 0d 78 2f 47 07 6c
# c0 bf 25 ed 9f 9a 9e e8 6e ff 5f b5 07 b6 57 05
# da 00 9a 62 ff 2f 0f e0 91 35 a4 cf 35 23 19 f2
# 1c 23 61 a4 85 1c 22 7f 75 68 7f 85 98 e8 0b 06
# fb 0e e5 32 73 d8 d6 7b 81 d2 d1 dc 84 84 09 41
# 82 ae 26 09 45 40 13 ae 3e 84 17 c8 2d 7f 07 38
# 3a a5 cc f9 ''', dtype=np.ubyte, count=100)

l = 1000
testbytes = np.fromstring(np.random.bytes(1000), dtype=np.ubyte, count=1000)


testbits = np.unpackbits(testbytes)
