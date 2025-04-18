import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
        self.output_min, self.output_max = output_limits
        
    def compute(self, setpoint, pv, dt):
        error = setpoint - pv
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        I = self.Ki * self.integral
        
        # Derivative term
        D = self.Kd * (error - self.prev_error) / dt
        
        # Save error for next derivative calculation
        self.prev_error = error
        
        # Compute output
        output = P + I + D
        
        # Apply output limits
        if self.output_min is not None and self.output_max is not None:
            output = np.clip(output, self.output_min, self.output_max)
            # Anti-windup: only integrate if not saturated
            if output != P + I + D:
                self.integral -= error * dt
        
        return output

def ziegler_nichols_tuning(system_func, initial_Kp=0.1, Kp_step=0.1, max_Kp=50.0,
                           setpoint=1.0, dt=0.01, min_time=5.0, max_time=30.0, 
                           output_limits=(None, None), plot_progress=False):
    """
    Perform Ziegler-Nichols tuning procedure to find Ku and Tu,
    then return PID parameters based on the classic Z-N rules.
    
    Args:
        system_func: Function that takes (control_signal, dt) and returns process variable
        initial_Kp: Starting proportional gain (default: 0.1)
        Kp_step: Increment step for Kp (default: 0.1)
        max_Kp: Maximum proportional gain to test (default: 10.0)
        setpoint: Desired setpoint (default: 1.0)
        dt: Time step for simulation (default: 0.01)
        min_time: Minimum simulation time to observe oscillations (default: 5.0)
        max_time: Maximum simulation time per test (default: 30.0)
        output_limits: Tuple of (min, max) output limits (default: (None, None))
        plot_progress: Whether to plot each tuning attempt (default: False)
    """
    # First find ultimate gain (Ku) and ultimate period (Tu)
    Ku, Tu = find_ultimate_gain_period(
        system_func=system_func,
        initial_Kp=initial_Kp,
        Kp_step=Kp_step,
        max_Kp=max_Kp,
        setpoint=setpoint,
        dt=dt,
        min_time=min_time,
        max_time=max_time,
        output_limits=output_limits,
        plot_progress=plot_progress
    )
    
    if Ku is None or Tu is None:
        raise ValueError("Failed to find sustained oscillations. Try adjusting parameters.")
    
    # Calculate PID parameters using Ziegler-Nichols rules
    return {
        'Ku': Ku,
        'Tu': Tu,
        'P': {'Kp': 0.5 * Ku, 'Ki': 0, 'Kd': 0},
        'PI': {'Kp': 0.45 * Ku, 'Ki': 0.54 * Ku / Tu, 'Kd': 0},
        'PID': {'Kp': 0.6 * Ku, 'Ki': 1.2 * Ku / Tu, 'Kd': 0.075 * Ku * Tu}
    }

def find_ultimate_gain_period(system_func, initial_Kp, Kp_step, max_Kp,
                             setpoint, dt, min_time, max_time,
                             output_limits, plot_progress=False):
    """Enhanced version with better oscillation detection"""
    Kp = initial_Kp
    best_oscillation = None
    
    while Kp <= max_Kp:
        time, pv, _ = simulate_p_controller(
            system_func, Kp, setpoint, dt, max_time, output_limits)
        
        peaks = find_peaks(pv)
        
        if len(peaks) >= 3:
            # Calculate stability metrics
            amplitudes = pv[peaks] - np.mean(pv)
            rel_std = np.std(amplitudes) / np.mean(amplitudes)
            periods = np.diff(time[peaks])
            period_std = np.std(periods)
            
            # Quality score for oscillations (higher is better)
            oscillation_quality = 1/(rel_std + 0.1) + 1/(period_std + 0.1)
            
            if best_oscillation is None or oscillation_quality > best_oscillation[0]:
                best_oscillation = (
                    oscillation_quality,
                    Kp,
                    np.mean(periods),
                    time,
                    pv
                )
        
        if plot_progress:
            plt.figure()
            plt.plot(time, pv)
            plt.title(f'Kp = {Kp:.2f}')
            plt.show()
        
        Kp += Kp_step
    
    if best_oscillation is not None:
        quality, Ku, Tu, time, pv = best_oscillation
        if quality > 5:  # Minimum quality threshold
            # if plot_progress:
            print(f"Found oscillations at Kp={Ku:.2f}, Tu={Tu:.2f} (quality={quality:.2f})")
            return Ku, Tu
    
    # Diagnostic plot if no oscillations found
    if plot_progress and best_oscillation is not None:
        quality, Ku, Tu, time, pv = best_oscillation
        plt.figure()
        plt.plot(time, pv)
        plt.title(f'Best attempt (Kp={Ku:.2f}, quality={quality:.2f})')
        plt.show()
    
    return None, None

def simulate_p_controller(system_func, Kp, setpoint, dt, max_time, output_limits):
    steps = int(max_time / dt)
    time = np.arange(0, max_time, dt)
    pv = np.zeros(steps)
    output = np.zeros(steps)
    
    controller = PIDController(Kp, 0, 0, output_limits)
    
    for i in range(1, steps):
        control = controller.compute(setpoint, pv[i-1], dt)
        pv[i] = system_func(control, dt)
        output[i] = control
    
    return time, pv, output

def find_peaks(signal):
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)
    return np.array(peaks)

# Example test system
class TestSystem:
    def __init__(self):
        self.pv = 0
        self.delay_buffer = [0] * 5  # Smaller delay for easier oscillation
        
    def update(self, control, dt):
        # Simpler system that's more likely to oscillate
        tau = 1.0  # Smaller time constant
        K = 2.0    # Higher gain
        L = 0.2    # Smaller delay
        delay_steps = int(L / dt)
        
        self.delay_buffer.append(control)
        delayed_control = self.delay_buffer.pop(0)
        self.pv += (K * delayed_control - self.pv) / tau * dt
        return self.pv

if __name__ == "__main__":
    system = TestSystem()
    def system_func(control, dt):
        return system.update(control, dt)
    
    try:
        zn_params = ziegler_nichols_tuning(
            system_func=system_func,
            initial_Kp=0.1,
            Kp_step=0.1,
            max_Kp=10.0,
            setpoint=1.0,
            dt=0.01,
            min_time=5.0,
            max_time=15.0,
            output_limits=(0, 10),
            plot_progress=0
        )
        
        print("Tuning successful! Parameters:")
        print(f"Ku: {zn_params['Ku']:.2f}, Tu: {zn_params['Tu']:.2f}")
        print(f"PID params: Kp={zn_params['PID']['Kp']:.2f}, Ki={zn_params['PID']['Ki']:.2f}, Kd={zn_params['PID']['Kd']:.2f}")
            
        # Test the tuned PID
        pid = PIDController(**zn_params['PID'], output_limits=(0, 10))
        system = TestSystem()
        
        dt = 0.01
        time = np.arange(0, 10, dt)
        pv = np.zeros_like(time)
        setpoint = np.ones_like(time)
        
        for i in range(1, len(time)):
            pv[i] = system.update(pid.compute(setpoint[i], pv[i-1], dt), dt)
        
        plt.figure(figsize=(10, 5))
        plt.plot(time, pv, label='Process Variable')
        plt.plot(time, setpoint, 'r--', label='Setpoint')
        plt.title('PID Control with Ziegler-Nichols Tuning')
        plt.legend()
        plt.grid()
        plt.show()
        
    except ValueError as e:
        print("Tuning failed. Try these adjustments:")
        print("1. Increase max_Kp (e.g., to 20.0)")
        print("2. Reduce system delay/time constant")
        print("3. Increase output limits")
        print("4. Use a different tuning method if oscillations can't be achieved")
        print(f"Error: {e}")
    
    print("========= DONE ============")
