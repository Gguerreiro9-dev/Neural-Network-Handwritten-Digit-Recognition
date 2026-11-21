package neural.activation;

import java.util.function.Function;

public class LeakyReLU implements IDifferentiableFunction {

    @Override
    public Function<Double, Double> fnc() {
        return x -> x > 0 ? x : 0.01 * x;
    }

    @Override
    public Function<Double, Double> derivative() {
        return x -> x > 0 ? 1.0 : 0.01;
    }
}