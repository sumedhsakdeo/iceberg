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
import org.apache.iceberg.types.Types;
import org.apache.iceberg.util.SerializableFunction;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;


public class TestEuclideanDistance {

    @Test
    public void testEuclideanDistanceFloat() {
        SerializableFunction<List<Float>, Float> binder = EuclideanDistance.get(List.of(1.0f, 2.0f, 3.0f), 3).bind(Types.FloatType.get());
        assertEquals(0.0f, binder.apply(List.of(1.0f, 2.0f, 3.0f)));
        assertEquals((float) Math.sqrt(27.0), binder.apply(List.of(4.0f, 5.0f, 6.0f)));
    }

    @Test
    public void testEuclideanDistanceDouble() {
        SerializableFunction<List<Double>, Double> binder = EuclideanDistance.get(List.of(1.0, 2.0, 3.0), 3).bind(Types.DoubleType.get());
        assertEquals(0.0, binder.apply(List.of(1.0, 2.0, 3.0)));
        assertEquals(Math.sqrt(27), binder.apply(List.of(4.0, 5.0, 6.0)));
    }

    @Test
    public void testEuclideanDistanceWithMismatchedDimension() {
        assertThatThrownBy(() -> EuclideanDistance.get(List.of(1.0f, 2.0f, 3.0f), 2))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("Vector size must be equal to dimension");
    }

    @Test
    public void euclideanDistanceFloatWithNullList() {
        NullPointerException exception = assertThrows(NullPointerException.class, () -> {
            EuclideanDistance.get(null, 3).bind(Types.FloatType.get());
        });
        assertThat(exception).hasMessageContaining("Vector cannot be null");
    }

    @Test
    public void euclideanDistanceDoubleWithNullList() {
        NullPointerException exception = assertThrows(NullPointerException.class, () -> {
            EuclideanDistance.get(null, 3).bind(Types.DoubleType.get());
        });
        assertThat(exception).hasMessageContaining("Vector cannot be null");
    }

    @Test
    public void euclideanDistanceWithInvalidType() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            EuclideanDistance.get(List.of(1.0, 2.0, 3.0), 3).bind(Types.StringType.get());
        });
        assertThat(exception).hasMessageContaining("Cannot compute euclidean distance for type");
    }

    @Test
    public void euclideanDistanceWithNegativeDimension() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            EuclideanDistance.get(List.of(1.0, 2.0, 3.0), -1).bind(Types.DoubleType.get());
        });
        assertThat(exception).hasMessageContaining("Dimension must be greater than 0");
    }

    @Test
    public void euclideanDistanceWithZeroDimension() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            EuclideanDistance.get(List.of(1.0, 2.0, 3.0), 0).bind(Types.DoubleType.get());
        });
        assertThat(exception).hasMessageContaining("Dimension must be greater than 0");
    }

}
