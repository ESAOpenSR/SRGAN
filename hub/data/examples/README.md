# Example data

`rgb-nir_batch.pt` is a serialized dictionary with normalized float32 tensors:
`lr` has shape `[4, 4, 128, 128]` (range 0.0147–0.61938) and `hr` has shape
`[4, 4, 512, 512]` (range 0.009–0.6496). The tensor example is already in the
historical `/10000` reflectance convention, so run it through the network-only
mode:

```bash
python examples/infer_tensor.py --preset RGB-NIR --input data/examples/rgb-nir_batch.pt \
  --raw --output rgb-nir_sr.pt
```

`dataset.zip` is a small example bundle for local experimentation. See the
matching [examples](../../examples/README.md) for the expected tensor and
GeoTIFF input layouts.

Sample imagery and data retain the original provider terms. They are not
relicensed by the repository MIT license; see [NOTICE.md](../../NOTICE.md).
