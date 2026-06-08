function [s_n_r, e] = SNR(xh, x)
% xh=quantized signal
% x=unquantized signal
% e=quantization error signal (optional)
% s_n_r=snr in dB
e = xh - x;
Ex = sum(x.^2);
Ee = sum(e.^2);
s_n_r = pow2db(Ex./Ee);
end