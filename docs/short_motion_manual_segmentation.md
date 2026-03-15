# 手动逐帧截取短动作（CSV -> 短CSV -> NPZ）

当长时序动作很难自动按事件切窗时，可使用 `scripts/csv_frame_segmenter.py` 逐帧手动截取。

## 1) 交互式逐帧标注并导出

```bash
python scripts/csv_frame_segmenter.py \
  --input_csv YOUR_LONG_MOTION.csv \
  --output_csv outputs/motions/jump_landing_short.csv \
  --interactive
```

交互命令：
- `n/p`：下一帧/上一帧
- `f k / b k`：前进/后退 k 帧
- `j idx`：跳转到 idx 帧
- `s/e`：将当前帧标为开始/结束
- `w`：写出当前 `[start,end]` 到输出 CSV
- `q`：退出

## 2) 非交互裁剪（已知帧范围）

```bash
python scripts/csv_frame_segmenter.py \
  --input_csv YOUR_LONG_MOTION.csv \
  --output_csv outputs/motions/jump_landing_short.csv \
  --start_frame 320 --end_frame 520
```

## 3) 转为 BeyondMimic 可用 NPZ

```bash
python scripts/csv_to_npz.py \
  --input_file outputs/motions/jump_landing_short.csv \
  --input_fps 30 \
  --output_name jump_landing_short \
  --output_fps 50
```

之后即可沿用原有 replay / 训练流程。
