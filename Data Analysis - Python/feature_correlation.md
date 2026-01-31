# Feature Correlation Analysis

**Total Features:** 192

**Dataset Shape:** (102, 193)

## Highly Correlated Feature Pairs (|r| > 0.9)

| Feature 1 | Feature 2 | Correlation |
|-----------|-----------|-------------|
| Gyro_X_std | Gyro_X_rms | 0.9996 |
| Gyro_X_std | Gyro_X_mad | 0.9690 |
| Gyro_X_iqr | Gyro_X_mad | 0.9430 |
| Gyro_X_iqr | Gyro_X_log_detector | 0.9709 |
| Gyro_X_rms | Gyro_X_mad | 0.9707 |
| Gyro_X_mad | Gyro_X_log_detector | 0.9372 |
| Gyro_Y_std | Gyro_Y_rms | 0.9999 |
| Gyro_Y_std | Gyro_Y_mad | 0.9737 |
| Gyro_Y_iqr | Gyro_Y_mad | 0.9196 |
| Gyro_Y_iqr | Gyro_Y_log_detector | 0.9528 |
| Gyro_Y_rms | Gyro_Y_mad | 0.9734 |
| Gyro_Y_mad | Gyro_Y_log_detector | 0.9569 |
| Gyro_Z_std | Gyro_Z_rms | 1.0000 |
| Gyro_Z_std | Gyro_Z_mad | 0.9878 |
| Gyro_Z_iqr | Gyro_Z_mad | 0.9246 |
| Gyro_Z_iqr | Gyro_Z_log_detector | 0.9548 |
| Gyro_Z_rms | Gyro_Z_mad | 0.9876 |
| Gyro_Z_mad | Gyro_Z_log_detector | 0.9380 |
| LinearAccel_X_mean | LinearAccel_X_median | 0.9364 |
| LinearAccel_X_std | LinearAccel_X_rms | 0.9771 |
| LinearAccel_X_std | LinearAccel_X_mad | 0.9664 |
| LinearAccel_X_std | RawAccel_X_std | 0.9941 |
| LinearAccel_X_std | RawAccel_X_mad | 0.9629 |
| LinearAccel_X_std | RawAccel_Z_std | 0.9049 |
| LinearAccel_X_max | RawAccel_X_max | 0.9073 |
| LinearAccel_X_iqr | LinearAccel_X_rms | 0.9039 |
| LinearAccel_X_iqr | LinearAccel_X_mad | 0.9352 |
| LinearAccel_X_iqr | LinearAccel_X_log_detector | 0.9140 |
| LinearAccel_X_iqr | RawAccel_X_iqr | 0.9850 |
| LinearAccel_X_iqr | RawAccel_X_rms | 0.9235 |
| LinearAccel_X_iqr | RawAccel_X_mad | 0.9298 |
| LinearAccel_X_skewness | RawAccel_X_skewness | 0.9670 |
| LinearAccel_X_kurtosis | RawAccel_X_kurtosis | 0.9955 |
| LinearAccel_X_rms | LinearAccel_X_mad | 0.9867 |
| LinearAccel_X_rms | RawAccel_X_std | 0.9821 |
| LinearAccel_X_rms | RawAccel_X_iqr | 0.9183 |
| LinearAccel_X_rms | RawAccel_X_mad | 0.9833 |
| LinearAccel_X_rms | RawAccel_Z_mad | 0.9009 |
| LinearAccel_X_mad | LinearAccel_Z_mad | 0.9044 |
| LinearAccel_X_mad | RawAccel_X_std | 0.9707 |
| LinearAccel_X_mad | RawAccel_X_iqr | 0.9503 |
| LinearAccel_X_mad | RawAccel_X_mad | 0.9954 |
| LinearAccel_X_mad | RawAccel_Z_mad | 0.9117 |
| LinearAccel_X_log_detector | RawAccel_X_iqr | 0.9280 |
| LinearAccel_Y_std | LinearAccel_Y_rms | 0.9999 |
| LinearAccel_Y_std | LinearAccel_Y_mad | 0.9822 |
| LinearAccel_Y_std | RawAccel_Y_std | 0.9946 |
| LinearAccel_Y_std | RawAccel_Y_rms | 0.9922 |
| LinearAccel_Y_std | RawAccel_Y_mad | 0.9817 |
| LinearAccel_Y_min | RawAccel_Y_min | 0.9480 |
| LinearAccel_Y_max | RawAccel_Y_max | 0.9450 |
| LinearAccel_Y_iqr | LinearAccel_Y_mad | 0.9485 |
| LinearAccel_Y_iqr | LinearAccel_Y_log_detector | 0.9607 |
| LinearAccel_Y_iqr | RawAccel_Y_iqr | 0.9634 |
| LinearAccel_Y_iqr | RawAccel_Y_mad | 0.9296 |
| LinearAccel_Y_kurtosis | RawAccel_Y_kurtosis | 0.9082 |
| LinearAccel_Y_rms | LinearAccel_Y_mad | 0.9827 |
| LinearAccel_Y_rms | RawAccel_Y_std | 0.9949 |
| LinearAccel_Y_rms | RawAccel_Y_rms | 0.9925 |
| LinearAccel_Y_rms | RawAccel_Y_mad | 0.9822 |
| LinearAccel_Y_mad | LinearAccel_Y_log_detector | 0.9095 |
| LinearAccel_Y_mad | RawAccel_Y_std | 0.9747 |
| LinearAccel_Y_mad | RawAccel_Y_iqr | 0.9319 |
| LinearAccel_Y_mad | RawAccel_Y_rms | 0.9686 |
| LinearAccel_Y_mad | RawAccel_Y_mad | 0.9906 |
| LinearAccel_Y_log_detector | RawAccel_Y_iqr | 0.9373 |
| LinearAccel_Z_std | LinearAccel_Z_rms | 0.9987 |
| LinearAccel_Z_std | LinearAccel_Z_mad | 0.9706 |
| LinearAccel_Z_std | RawAccel_Z_std | 0.9945 |
| LinearAccel_Z_std | RawAccel_Z_rms | 0.9465 |
| LinearAccel_Z_std | RawAccel_Z_mad | 0.9735 |
| LinearAccel_Z_min | RawAccel_Z_min | 0.9803 |
| LinearAccel_Z_max | RawAccel_Z_max | 0.9849 |
| LinearAccel_Z_iqr | LinearAccel_Z_mad | 0.9402 |
| LinearAccel_Z_iqr | LinearAccel_Z_log_detector | 0.9529 |
| LinearAccel_Z_iqr | RawAccel_Z_iqr | 0.9580 |
| LinearAccel_Z_iqr | RawAccel_Z_mad | 0.9124 |
| LinearAccel_Z_skewness | RawAccel_Z_skewness | 0.9539 |
| LinearAccel_Z_kurtosis | RawAccel_Z_kurtosis | 0.9748 |
| LinearAccel_Z_rms | LinearAccel_Z_mad | 0.9737 |
| LinearAccel_Z_rms | RawAccel_Z_std | 0.9931 |
| LinearAccel_Z_rms | RawAccel_Z_rms | 0.9516 |
| LinearAccel_Z_rms | RawAccel_Z_mad | 0.9756 |
| LinearAccel_Z_mad | LinearAccel_Z_log_detector | 0.9187 |
| LinearAccel_Z_mad | RawAccel_X_mad | 0.9034 |
| LinearAccel_Z_mad | RawAccel_Z_std | 0.9559 |
| LinearAccel_Z_mad | RawAccel_Z_iqr | 0.9393 |
| LinearAccel_Z_mad | RawAccel_Z_rms | 0.9076 |
| LinearAccel_Z_mad | RawAccel_Z_mad | 0.9899 |
| LinearAccel_Z_log_detector | RawAccel_Z_iqr | 0.9330 |
| LinearAccel_Z_log_detector | RawAccel_Z_mad | 0.9032 |
| Magnetometer_X_mean | Magnetometer_X_median | 0.9650 |
| Magnetometer_X_std | Magnetometer_X_iqr | 0.9420 |
| Magnetometer_X_std | Magnetometer_X_mad | 0.9932 |
| Magnetometer_X_iqr | Magnetometer_X_mad | 0.9714 |
| Magnetometer_Y_std | Magnetometer_Y_min | -0.9361 |
| Magnetometer_Y_std | Magnetometer_Y_max | 0.9152 |
| Magnetometer_Y_std | Magnetometer_Y_iqr | 0.9885 |
| Magnetometer_Y_std | Magnetometer_Y_rms | 0.9970 |
| Magnetometer_Y_std | Magnetometer_Y_mad | 0.9953 |
| Magnetometer_Y_std | Magnetometer_Y_log_detector | 0.9534 |
| Magnetometer_Y_min | Magnetometer_Y_max | -0.9083 |
| Magnetometer_Y_min | Magnetometer_Y_iqr | -0.9011 |
| Magnetometer_Y_min | Magnetometer_Y_rms | -0.9319 |
| Magnetometer_Y_min | Magnetometer_Y_mad | -0.9095 |
| Magnetometer_Y_max | Magnetometer_Y_rms | 0.9247 |
| Magnetometer_Y_iqr | Magnetometer_Y_rms | 0.9825 |
| Magnetometer_Y_iqr | Magnetometer_Y_mad | 0.9898 |
| Magnetometer_Y_iqr | Magnetometer_Y_log_detector | 0.9486 |
| Magnetometer_Y_rms | Magnetometer_Y_mad | 0.9903 |
| Magnetometer_Y_rms | Magnetometer_Y_log_detector | 0.9548 |
| Magnetometer_Y_mad | Magnetometer_Y_log_detector | 0.9714 |
| Magnetometer_Z_std | Magnetometer_Z_rms | 0.9402 |
| Magnetometer_Z_std | Magnetometer_Z_mad | 0.9373 |
| Magnetometer_Z_iqr | Magnetometer_Z_mad | 0.9131 |
| RawAccel_X_mean | RawAccel_X_median | 0.9894 |
| RawAccel_X_std | RawAccel_X_mad | 0.9746 |
| RawAccel_X_std | RawAccel_Z_std | 0.9001 |
| RawAccel_X_iqr | RawAccel_X_rms | 0.9075 |
| RawAccel_X_iqr | RawAccel_X_mad | 0.9549 |
| RawAccel_X_mad | RawAccel_Z_mad | 0.9113 |
| RawAccel_Y_std | RawAccel_Y_rms | 0.9977 |
| RawAccel_Y_std | RawAccel_Y_mad | 0.9856 |
| RawAccel_Y_iqr | RawAccel_Y_mad | 0.9408 |
| RawAccel_Y_iqr | RawAccel_Y_log_detector | 0.9030 |
| RawAccel_Y_rms | RawAccel_Y_mad | 0.9817 |
| RawAccel_Z_mean | RawAccel_Z_median | 0.9533 |
| RawAccel_Z_std | RawAccel_Z_rms | 0.9575 |
| RawAccel_Z_std | RawAccel_Z_mad | 0.9708 |
| RawAccel_Z_iqr | RawAccel_Z_mad | 0.9445 |
| RawAccel_Z_rms | RawAccel_Z_mad | 0.9345 |
| Quat_W_mean | Quat_W_median | 0.9465 |
| Quat_W_std | Quat_W_iqr | 0.9479 |
| Quat_W_std | Quat_W_mad | 0.9844 |
| Quat_W_iqr | Quat_W_mad | 0.9693 |
| Quat_W_rms | Quat_X_rms | -0.9296 |
| Quat_X_mean | Quat_X_median | 0.9301 |
| Quat_X_std | Quat_X_mad | 0.9803 |
| Quat_X_iqr | Quat_X_mad | 0.9528 |
| Quat_Y_mean | Quat_Y_median | 0.9417 |
| Quat_Y_std | Quat_Y_iqr | 0.9388 |
| Quat_Y_std | Quat_Y_mad | 0.9829 |
| Quat_Y_iqr | Quat_Y_mad | 0.9672 |
| Quat_Y_rms | Quat_Z_rms | -0.9003 |
| Quat_Z_mean | Quat_Z_median | 0.9376 |
| Quat_Z_std | Quat_Z_iqr | 0.9121 |
| Quat_Z_std | Quat_Z_mad | 0.9841 |
| Quat_Z_iqr | Quat_Z_mad | 0.9601 |

## Feature Statistics

| Statistic | Value |
|-----------|-------|
| Mean Correlation | 0.0349 |
| Max Correlation | 1.0000 |
| Min Correlation | -0.9361 |

## Recommendations

- Features with correlation > 0.9 may be redundant
- Consider feature selection techniques in the ML pipeline
- Use dimensionality reduction (PCA) if needed
