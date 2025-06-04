/*
 *
 *  * Licensed to the Apache Software Foundation (ASF) under one
 *  * or more contributor license agreements.  See the NOTICE file
 *  * distributed with this work for additional information
 *  * regarding copyright ownership.  The ASF licenses this file
 *  * to you under the Apache License, Version 2.0 (the
 *  * "License"); you may not use this file except in compliance
 *  * with the License.  You may obtain a copy of the License at
 *  *
 *  *   http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  * Unless required by applicable law or agreed to in writing,
 *  * software distributed under the License is distributed on an
 *  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  * KIND, either express or implied.  See the License for the
 *  * specific language governing permissions and limitations
 *  * under the License.
 *
 */

package org.apache.iceberg.transforms;

import java.util.List;
import java.util.function.Function;
import org.apache.iceberg.expressions.BoundPredicate;
import org.apache.iceberg.expressions.UnboundPredicate;
import org.apache.iceberg.relocated.com.google.common.base.Objects;
import org.apache.iceberg.relocated.com.google.common.base.Preconditions;
import org.apache.iceberg.types.Type;
import org.apache.iceberg.types.Types;
import org.apache.iceberg.util.SerializableFunction;

public class EuclideanDistance<T> implements Transform<List<T>, T>, Function<List<T>, T> {
    static <T> EuclideanDistance<T> get(List<T> vector, int dimension) {
        Preconditions.checkNotNull(vector, "Vector cannot be null");
        Preconditions.checkArgument(dimension > 0, "Dimension must be greater than 0");
        Preconditions.checkArgument(vector.size() == dimension, "Vector size must be equal to dimension");
        return new EuclideanDistance<>(vector, dimension);
    }

    private final List<T> vector;
    private final int dimension;

    private EuclideanDistance(List<T> vector, int dimension) {
        this.vector = vector;
        this.dimension = dimension;
    }

    public List<T> vector() {
        return vector;
    }

    public int dimension() {
        return dimension;
    }

    @Override
    public SerializableFunction<List<T>, T> bind(Type type) {
        Preconditions.checkArgument(canTransform(type), "Cannot compute euclidean distance for type: %s", type);
        return value -> {
            Preconditions.checkArgument(vector.size() == dimension, "Vector size must be equal to dimension");
            double sum = 0;
            for (int i = 0; i < dimension; i++) {
                double diff = ((Number) value.get(i)).doubleValue() - ((Number) EuclideanDistance.this.vector.get(i)).doubleValue();
                sum += diff * diff;
            }
            switch (type.typeId()) {
                case FLOAT:
                    return (T) Float.valueOf((float) Math.sqrt(sum));
                case DOUBLE:
                    return (T) Double.valueOf(Math.sqrt(sum));
                default:
                    throw new IllegalArgumentException("Invalid type: " + type);
            }
        };
    }

    @Override
    public T apply(List<T> value) {
        throw new UnsupportedOperationException(
                "apply(value) is deprecated, use bind(Type).apply(value)");
    }

    @Override
    public boolean canTransform(Type type) {
        return type.typeId() == Type.TypeID.FLOAT || type.typeId() == Type.TypeID.DOUBLE;
    }

    @Override
    public Type getResultType(Type sourceType) {
        Preconditions.checkArgument(sourceType.typeId() != Type.TypeID.LIST, "Invalid source type: %s", sourceType);
        switch (sourceType.asNestedType().asListType().elementType().typeId()) {
            case FLOAT:
                return Types.FloatType.get();
            case DOUBLE:
                return Types.DoubleType.get();
            default:
                throw new IllegalArgumentException("Invalid source type: " + sourceType);
        }
    }

    @Override
    public UnboundPredicate<T> project(String name, BoundPredicate<List<T>> predicate) {
        return get(vector, dimension).project(name, predicate);
    }

    @Override
    public UnboundPredicate<T> projectStrict(String name, BoundPredicate<List<T>> predicate) {
        return get(vector, dimension).projectStrict(name, predicate);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        } else if (!(o instanceof EuclideanDistance)) {
            return false;
        }

        EuclideanDistance<?> that = (EuclideanDistance<?>) o;
        return dimension == that.dimension() && vector.equals(that.vector());
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(vector, dimension);
    }

    @Override
    public String toString() {
        return "euclidean_distance(ref(name=" + vector + "), " + vector + ", " + dimension + ")";
    }

}
