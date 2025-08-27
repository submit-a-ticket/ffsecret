# FFSecret

Fourier-domain, spread-spectrum image watermarking.

## Install
```bash
python -m pip install -r requirements.txt
```

## CLI usage
```bash
# Embed a secret string
python -m ffsecret embed cover.jpg stego.png "Hello World" --key 12345

# Embed bytes from a file
python -m ffsecret embed cover.jpg stego.png @secret.bin --key 4242

# Extract
python -m ffsecret extract stego.png --key 4242 -o recovered.bin
```

## Parameters
* `--alpha` – embedding strength (default `0.1`). Lower = less visible but less robust.
* `--spread` – number of coefficient pairs per bit (default `5`). Higher = better robustness, lower capacity.
Presets (choose one with `--preset`)

| preset   | alpha | spread | visibility | typical use |
|----------|-------|--------|-------------|--------------|
| low      | 0.05  | 7      | invisible   | archival where no processing expected |
| medium   | 0.10  | 5      | imperceptible | everyday photos (default) |
| high     | 0.25  | 3      | faint texture | survives one JPEG re-save / social-media upload |
| extreme  | 0.40  | 1      | visible speckle | forensic, survives heavy recompression |

## How it works
FFSecret hides a small payload in the Fourier domain of the image luminance:
1. The cover is converted to YCbCr and only the Y (brightness) channel is processed – colour manipulations won’t hurt the watermark.
2. A 2-D FFT is taken and shifted so the DC term sits in the centre.
3. We keep only the mid-frequency ring (≈15–45 % of the Nyquist radius) and only the upper-left half-plane, giving one unique member of every conjugate pair.
4. A pseudorandom permutation (seeded by the secret `--key`) picks coefficient pairs.  Each payload bit is repeated `spread` times (spread-spectrum).
5. For every selected pair (k, −k):
   • bit 1 → ensure |F(k)| > |F(−k)|  
   • bit 0 → ensure |F(k)| < |F(−k)|  
   Then both magnitudes are nudged by ±`alpha` (e.g. ±25 %).
6. An inverse FFT brings the image back to the spatial domain; we merge the untouched Cb/Cr channels and save.
7. Extraction does the same walk, takes a majority vote over each `spread` block, reads a 16-bit length header and Reed–Solomon-decodes the payload.

Presets
* **default** – `alpha 0.1`, `spread 5` → imperceptible on typical photos.
* **robust**  – `alpha 0.25`, `spread 3` → survives very flat graphics; may leave a faint texture.

Typical capacity is ≈ 128–256 bytes per megapixel with the default settings.

## Library usage
```python
from ffsecret import embed, extract

# Embed
embed("cover.jpg", "stego.png", b"top-secret", key=9876)

# Extract
payload = extract("stego.png", key=9876)
print(payload)
```

## Examples
Original
![original](https://github.com/submit-a-ticket/ffsecret/blob/main/initial.png?raw=true)

Watermark embeded (extreme preset with crop safe)
![original](https://github.com/submit-a-ticket/ffsecret/blob/main/stego.png?raw=true)
