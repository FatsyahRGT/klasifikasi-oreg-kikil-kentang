const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput'); // Tambahkan input type file hidden di HTML
const stepsDiv = document.getElementById('steps');
const resultDiv = document.getElementById('result');

// Memicu klik input file saat dropzone diklik
dropzone.addEventListener('click', () => fileInput.click());

// Menangani perubahan file (saat user memilih foto)
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    await processImage(file);
});

async function processImage(file) {
    // 1. Persiapkan Animasi UI
    resultDiv.style.display = 'none';
    stepsDiv.style.display = 'block';
    stepsDiv.innerHTML = '<li>Memuat gambar...</li>';
    
    // Menambahkan ilusi proses agar terlihat canggih
    setTimeout(() => stepsDiv.innerHTML += '<li>Mengekstrak Histogram Warna RGB...</li>', 600);
    setTimeout(() => stepsDiv.innerHTML += '<li>Menghitung matriks tekstur GLCM...</li>', 1200);
    setTimeout(() => stepsDiv.innerHTML += '<li>Mengklasifikasi dengan K-NN...</li>', 1800);

    // 2. Siapkan data untuk dikirim ke Python
    const formData = new FormData();
    formData.append('file', file);

    try {
        // 3. Kirim gambar ke server backend (Flask)
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // 4. Tampilkan Hasil setelah animasi selesai (delay buatan agar sinkron dengan UI)
        setTimeout(() => {
            stepsDiv.style.display = 'none';
            resultDiv.style.display = 'block';
            
            if(data.error) {
                resultDiv.innerHTML = `<span style="color:red;">Error: ${data.error}</span>`;
            } else {
                resultDiv.innerHTML = `<h3>Hasil Prediksi: ${data.result}</h3>`;
            }
        }, 2200);

    } catch (error) {
        stepsDiv.style.display = 'none';
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = '<span style="color:red;">Koneksi ke server gagal.</span>';
    }


async function processImage(file) {
    // 1. Tampilkan preview gambar
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    imagePreview.src = URL.createObjectURL(file);
    previewContainer.style.display = 'block';

    // Sembunyikan dropzone agar UI lebih rapi saat memproses
    dropzone.style.display = 'none';

    // 2. Persiapkan Animasi UI
    resultDiv.style.display = 'none';
    stepsDiv.style.display = 'block';
    stepsDiv.innerHTML = '<li>Mengirim gambar ke server...</li>';
    
    setTimeout(() => stepsDiv.innerHTML += '<li>Mengekstrak Histogram Warna RGB...</li>', 600);
    setTimeout(() => stepsDiv.innerHTML += '<li>Menghitung matriks tekstur GLCM...</li>', 1200);
    setTimeout(() => stepsDiv.innerHTML += '<li>Mengklasifikasi dengan K-NN...</li>', 1800);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();

        setTimeout(() => {
            stepsDiv.style.display = 'none';
            resultDiv.style.display = 'block';
            
            // Tampilkan kembali dropzone untuk unggah gambar berikutnya
            dropzone.style.display = 'block';
            dropzone.querySelector('p').innerText = "Coba gambar lain?";
            
            if(data.error) {
                resultDiv.innerHTML = `<span style="color:red;">Error: ${data.error}</span>`;
            } else {
                resultDiv.innerHTML = `<h3>Prediksi: ${data.result}</h3>`;
            }
        }, 2200);

    } catch (error) {
        // ... (penanganan error sama)
    }
}
}