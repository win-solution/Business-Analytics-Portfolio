# Membuat list dan dictionary untuk menyimpan informasi barang yang tersedia
etalase_toko = [
    {
    "jenis" : "sabun batang",
    "merk" : "Lifebuoy Mildcare",
    "harga" : 3000,
    "stok" : 20
    },
    {
    "jenis" : "sabun batang",
    "merk" : "Nuvo Active Clean",
    "harga" : 2500,
    "stok" : 30
    },
    {
    "jenis" : "shampo botol",
    "merk" : "Sunsilk Black Shine",
    "harga" : 18000,
    "stok" : 20
    },
    {
    "jenis" : "shampo botol",
    "merk" : "Pantene Anti Ketombe",
    "harga" : 19000,
    "stok" : 15
    },
    {
    "jenis" : "pasta gigi",
    "merk" : "Pepsodent Complete 8",
    "harga" : 11000,
    "stok" : 10
    },
    {
    "jenis" : "pasta gigi",
    "merk" : "Ciptadent Maxi Herbal",
    "harga" : 9000,
    "stok" : 10
    }
]

# Fungsi untuk menampilkan daftar barang yang tersedia di etalase (Read Menu)
def daftar_barang():
    print('''
Kode etalase sabun batang = 1
Kode etalase shampo botol = 2
Kode etalase pasta gigi = 3
Kode yang menampilkan semua etalase = 4
Kode untuk kembali ke menu utama = 5
    ''')
    etalase = input('Silakan pilih kode yang Anda inginkan (kode 1, 2, 3, 4, atau 5): ')
    if(etalase == '1'):
        print(('*')*33+' Etalase Sabun Batang'+('*')*33)
        print('Index\t| Jenis\t\t  | Merk  \t\t\t  | Harga\t\t| Stok')
        for item in range(len(etalase_toko)):
            if(etalase_toko[item]['jenis']=='sabun batang'):
                print('{}.\t| {}\t  | {}    \t  | Rp {}/unit\t| {}'.
                      format(item, etalase_toko[item]['jenis'], etalase_toko[item]['merk'], etalase_toko[item]['harga'], etalase_toko[item]['stok']))
        daftar_barang()
    elif(etalase == '2'):
        print(('*')*33+' Etalase Shampo Botol'+('*')*33)
        print('Index\t| Jenis\t\t  | Merk  \t\t\t  | Harga\t\t| Stok')
        for item in range(len(etalase_toko)):
            if(etalase_toko[item]['jenis']=='shampo botol'):
                print('{}.\t| {}\t  | {}    \t  | Rp {}/unit\t| {}'.
                      format(item, etalase_toko[item]['jenis'], etalase_toko[item]['merk'], etalase_toko[item]['harga'], etalase_toko[item]['stok']))
        daftar_barang()
    elif(etalase == '3'):
        print(('*')*33+' Etalase Pasta Gigi '+('*')*33)
        print('Index\t| Jenis\t\t  | Merk  \t\t\t  | Harga\t\t| Stok')
        for item in range(len(etalase_toko)):
            if(etalase_toko[item]['jenis']=='pasta gigi'):
                print('{}.\t| {}\t  | {}    \t  | Rp {}/unit\t| {}'.
                      format(item, etalase_toko[item]['jenis'], etalase_toko[item]['merk'], etalase_toko[item]['harga'], etalase_toko[item]['stok']))
        daftar_barang()
    elif(etalase == '4'):
        print(('*')*33+' Etalase Semua Barang '+('*')*33)
        print('Index\t| Jenis\t\t  | Merk  \t\t\t  | Harga\t\t| Stok')
        for item in range(len(etalase_toko)):
            print('{}.\t| {}\t  | {}    \t  | Rp {}/unit\t| {}'.
                  format(item, etalase_toko[item]['jenis'], etalase_toko[item]['merk'], etalase_toko[item]['harga'], etalase_toko[item]['stok']))
        daftar_barang()
    elif(etalase == '5'):
        menu_utama()
    else:
        print('\nKode yang Anda masukkan salah, silakan masukkan kode etalase yang benar')
        daftar_barang()

# Fungsi untuk menambahkan barang baru ke dalam etalase (Create Menu)
def menambah_barang():
    print()
    while(True):
        validation = input('Ketik 1 jika anda ingin lanjut menambahkan barang, ketik 2 jika ingin kembali ke menu utama: ')
        if(validation == '1'):
            print()
            merk_tambah = input('Masukkan merk yang akan ditambahkan untuk mengecek apakah barang sudah ada di etalase: ')
            for item in range(len(etalase_toko)):
                if(merk_tambah == etalase_toko[item]["merk"]):
                    print('\nMaaf, barang sudah ada di etalase, silakan masukkan barang lainnya')
                    menambah_barang()
                    break
            else:
                while(True):
                    print('\nMerk yang dimasukkan belum ada di etalase, Anda bisa menambahkan merk tersebut')
                    print('''
                    Kode sabun batang = 1
                    Kode shampo botol = 2
                    Kode pasta gigi = 3
                    ''')
                    jenis_tambah = input('Masukkan kode jenis barang (1, 2, atau 3): ')
                    if(jenis_tambah == '1'):
                        harga_tambah = int(input('Masukkan harga barang: '))
                        stok_tambah = int(input('Masukkan stok barang: '))
                        jenis_tambah1 = "sabun batang"
                        while(True):
                            validate_tambah = input('Apakah yakin untuk menambahkan barang ini (ya/tidak)?: ')
                            if(validate_tambah == 'ya'):
                                etalase_toko.append({
                                    "jenis" : jenis_tambah1,
                                    "merk" : merk_tambah,
                                    "harga" : harga_tambah,
                                    "stok" : stok_tambah
                                    })
                                print('\nBarang baru telah berhasil ditambahkan')
                                daftar_barang()
                                break
                            elif(validate_tambah == 'tidak'):
                                menambah_barang()
                            else:
                                print('Anda hanya bisa memilih "ya" atau "tidak"')
                                continue
                            break
                    elif(jenis_tambah == '2'):
                        harga_tambah = int(input('Masukkan harga barang: '))
                        stok_tambah = int(input('Masukkan stok barang: '))
                        jenis_tambah1 = "shampo botol"
                        while(True):
                            validate_tambah = input('Apakah yakin untuk menambahkan barang ini (ya/tidak)?: ')
                            if(validate_tambah == 'ya'):
                                etalase_toko.append({
                                    "jenis" : jenis_tambah1,
                                    "merk" : merk_tambah,
                                    "harga" : harga_tambah,
                                    "stok" : stok_tambah
                                    })
                                print('\nBarang baru telah berhasil ditambahkan')
                                daftar_barang()
                                break
                            elif(validate_tambah == 'tidak'):
                                menambah_barang()
                            else:
                                print('Anda hanya bisa memilih "ya" atau "tidak"')
                                continue
                            break
                    elif(jenis_tambah == '3'):
                        harga_tambah = int(input('Masukkan harga barang: '))
                        stok_tambah = int(input('Masukkan stok barang: '))
                        jenis_tambah1 = "pasta gigi"
                        while(True):
                            validate_tambah = input('Apakah yakin untuk menambahkan barang ini (ya/tidak)?: ')
                            if(validate_tambah == 'ya'):
                                etalase_toko.append({
                                    "jenis" : jenis_tambah1,
                                    "merk" : merk_tambah,
                                    "harga" : harga_tambah,
                                    "stok" : stok_tambah
                                    })
                                print('\nBarang baru telah berhasil ditambahkan')
                                daftar_barang()
                                break
                            elif(validate_tambah == 'tidak'):
                                menambah_barang()
                            else:
                                print('Anda hanya bisa memilih "ya" atau "tidak"')
                                continue
                            break        
                    else:
                        print('\nKode yang Anda masukkan salah, silakan masukkan kode jenis yang tersedia')
                        continue
                    break
        elif(validation == '2'):
            menu_utama()
            break
        else:
            print('Anda hanya bisa memilih kode 1 atau 2')
            continue
        break

# Fungsi untuk melakukan update pada informasi barang yang tersedia di etalase (Update Menu)
def update_barang():
    while(True):
        print()
        validation = input('Ketik 1 jika anda ingin lanjut untuk update barang, ketik 2 jika ingin kembali ke menu utama: ')
        if(validation == '1'):   
            print(('*')*33+' Etalase Semua Barang '+('*')*33)
            print('Index\t| Jenis\t\t  | Merk  \t\t\t  | Harga\t\t| Stok')
            for item in range(len(etalase_toko)):
                print('{}.\t| {}\t  | {}    \t  | Rp {}/unit\t| {}'.
                      format(item, etalase_toko[item]['jenis'], etalase_toko[item]['merk'], etalase_toko[item]['harga'], etalase_toko[item]['stok']))
            index = int(input('Masukkan index barang yang ingin diubah (angkanya saja): '))
            print(('*')*33+' Barang yang Dipilih '+('*')*33)
            if index in range(len(etalase_toko)):
                print()
                print('Index\t| Jenis\t\t  | Merk  \t\t\t  | Harga\t\t| Stok')
                print('{}.\t| {}\t  | {}    \t  | Rp {}/unit\t| {}'.
                      format(index, etalase_toko[index]['jenis'], etalase_toko[index]['merk'], etalase_toko[index]['harga'], etalase_toko[index]['stok']))
                while(True):
                    print()
                    validate_update = input('Apakah lanjut untuk melakukan update (ya/tidak)?: ')
                    if(validate_update == 'ya'):
                        print('''
                        Kode update jenis = 1
                        Kode update merk = 2
                        Kode update harga = 3
                        Kode update stok = 4
                        ''')
                        while(True):
                            update = input('Silakan pilih kode update yang ingin dilakukan: ')
                            if(update == '1'):
                                jenis_baru = input('Masukkan jenis baru yang diinginkan: ')
                                etalase_toko[index]['jenis'] = jenis_baru
                                print('\nJenis dari barang yang dipilih telah berhasil diubah')
                                daftar_barang()
                                break
                            elif(update == '2'):
                                merk_baru = input('Masukkan merk baru yang diinginkan: ')
                                etalase_toko[index]['merk'] = merk_baru
                                print('\nMerk dari barang yang dipilih telah diubah')
                                daftar_barang()
                                break
                            elif(update == '3'):
                                harga_baru = int(input('Masukkan harga baru yang diinginkan: '))
                                etalase_toko[index]['harga'] = harga_baru
                                print('\nHarga dari barang yang dipilih telah diubah')
                                daftar_barang()
                                break
                            elif(update == '4'):
                                stok_baru = int(input('Masukkan stok baru yang diinginkan: '))
                                etalase_toko[index]['stok'] = stok_baru
                                print('\nStok dari barang yang dipilih telah diubah')
                                daftar_barang()
                                break
                            else:
                                print('\nKode yang Anda masukkan salah, silakan masukkan kode update yang tersedia')
                    elif(validate_update == 'tidak'):   
                        update_barang()
                    else:
                        print('Anda hanya bisa memilih "ya" atau "tidak"')
                        continue
                    break
            else:
                print("\nMaaf, barang tidak ditemukan, silakan coba kembali")
                update_barang()
        elif(validation == '2'):
            menu_utama()
            break
        else:
            print('Anda hanya bisa memilih kode 1 atau 2')
            continue
        break

# Fungsi untuk menghapus barang dari etalase (Delete Menu)
def menghapus_barang():
    while(True):
        print()
        validation = input('Ketik 1 jika anda ingin lanjut menghapus barang, ketik 2 jika ingin kembali ke menu utama: ')
        if(validation == '1'):
            print(('*')*33+' Etalase Semua Barang '+('*')*33)
            print('Index\t| Jenis\t\t  | Merk  \t\t\t  | Harga\t\t| Stok')
            for item in range(len(etalase_toko)):
                print('{}.\t| {}\t  | {}    \t  | Rp {}/unit\t| {}'.
                      format(item, etalase_toko[item]['jenis'], etalase_toko[item]['merk'], etalase_toko[item]['harga'], etalase_toko[item]['stok']))
            index = int(input('Masukkan index barang yang ingin dihapus (angkanya saja): '))
            if index in range(len(etalase_toko)):
                while(True):
                    validate_hapus = input('Apakah yakin untuk menghapus barang ini (ya/tidak)?: ')
                    if(validate_hapus == 'ya'):
                        del etalase_toko[index]
                        print('\nData barang yang dipilih telah berhasil dihapus')
                        daftar_barang()
                        break
                    elif(validate_hapus == 'tidak'):   
                        menghapus_barang()
                    else:
                        print('Anda hanya bisa memilih "ya" atau "tidak"')
                        continue
                    break
            else:
                print("\nMaaf, barang tidak ditemukan, silakan coba kembali")
                menghapus_barang()
        elif(validation == '2'):
            menu_utama()
            break
        else:
            print('Anda hanya bisa memilih kode 1 atau 2')
            continue
        break

# Fungsi untuk membeli barang dari etalase toko (Additional Menu)
keranjang = []
def penjualan_barang():
    while(True):
        print()
        validation = input('Ketik 1 jika anda ingin lanjut membeli barang, ketik 2 jika ingin kembali ke menu utama: ')
        if(validation == '1'):
            print(('*')*33+' Etalase Semua Barang '+('*')*33)
            print('Index\t| Jenis\t\t  | Merk  \t\t\t  | Harga\t\t| Stok')
            for item in range(len(etalase_toko)):
                print('{}.\t| {}\t  | {}    \t  | Rp {}/unit\t| {}'.
                      format(item, etalase_toko[item]['jenis'], etalase_toko[item]['merk'], etalase_toko[item]['harga'], etalase_toko[item]['stok']))
            while(True):
                while(True):
                    index = int(input('Masukkan index barang yang dibeli = '))
                    if index in range(len(etalase_toko)):
                        break
                    else:
                        print('\nIndex barang yang Anda masukkan tidak tersedia, silakan masukkan index yang tersedia\n')
                qty = int(input('Masukkan jumlah barang yang dibeli = '))
                if(qty > etalase_toko[index]['stok']):
                    print('\nStok {} tidak cukup, stok {} tinggal {}'.format(etalase_toko[index]['merk'], etalase_toko[index]['merk'], etalase_toko[index]['stok']))
                    penjualan_barang()
                else:
                    keranjang.append({
                        "index" : index,
                        "merk": etalase_toko[index]['merk'], 
                        "qty": qty, 
                        "harga": etalase_toko[index]['harga'], 
                        })
                print()
                print(('*')*23+' Isi Keranjang '+('*')*23)
                print('Index\t| Merk  \t\t  | Qty\t| Harga')
                for item in keranjang:
                    print('{}.\t| {}\t  | {}\t| Rp {}/unit'.format(item['index'], item['merk'], item['qty'], item['harga']))
                print()
                while(True):
                    validate_jual = input('Apakah ada tambahan barang lain yang dibeli (ya/tidak)?: ')
                    if(validate_jual == 'tidak'):
                        total_harga = 0
                        print()
                        print(('*')*23+' Barang yang Dibeli '+('*')*23)
                        print('Merk\t\t\t  | Qty\t  | Harga/Unit\t | Total Harga')
                        for item in keranjang:
                            print('{}\t  | {} \t  | Rp {}\t | Rp {}'.format(item['merk'], item['qty'], item['harga'], item['qty']*item['harga']))
                            total_harga += item['qty']*item['harga']    
                        while(True):
                            print('\nTotal yang harus dibayar = Rp {}'.format(total_harga))
                            jumlah_bayar = int(input('Masukkan jumlah pembayaran = Rp '))
                            if(jumlah_bayar > total_harga):
                                kembalian = jumlah_bayar - total_harga
                                print('Terima kasih\nUang kembalian anda = Rp {}'.format(kembalian))
                                for item in keranjang:
                                    etalase_toko[item['index']]['stok'] -= item['qty']
                                keranjang.clear()
                                menu_utama()
                            elif(jumlah_bayar == total_harga):
                                print('Terima kasih')
                                for item in keranjang:
                                    etalase_toko[item['index']]['stok'] -= item['qty']
                                keranjang.clear()
                                menu_utama()
                            else:
                                kekurangan = total_harga - jumlah_bayar
                                print('Uang Anda kurang sebesar Rp {}'.format(kekurangan))
                                continue
                            break
                    elif(validate_jual == 'ya'):
                        penjualan_barang()
                    else:
                        print('Anda hanya bisa memilih "ya" atau "tidak"')
                        continue
                    break
                break
        elif(validation == '2'):
            menu_utama()
            break
        else:
            print('Anda hanya bisa memilih kode 1 atau 2')
            continue
        break
            
# Fungsi menu utama untuk masuk ke program etalase toko (Main Menu)
def menu_utama():
    while (True):
        main_menu = input('''
                Selamat datang di Toko Ceria!

        **************** Menu Utama ******************
        1. Menampilkan isi etalase
        2. Menambah barang baru ke etalase
        3. Update informasi barang yang ada di etalase
        4. Menghapus barang dari etalase
        5. Membeli barang yang ada di etalase
        6. Keluar dari program menu utama

        Masukkan angka menu yang ingin dijalankan: ''')

        if(main_menu == '1'):
            daftar_barang()
        elif(main_menu == '2'):
            menambah_barang()
        elif(main_menu == '3'):
            update_barang()
        elif(main_menu == '4'):
            menghapus_barang()
        elif(main_menu == '5'):
            penjualan_barang()
        elif(main_menu == '6'):
            break
        else:
            print('''
        Angka menu yang Anda masukkan salah, silakan coba kembali''')
            continue
        break

while(True):
    menu_utama()
    break

    









