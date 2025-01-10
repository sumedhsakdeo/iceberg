/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.iceberg.actions;

import java.util.Map;
import org.apache.iceberg.StatisticsFile;

public interface ComputeTableEmbeddings
    extends Action<ComputeTableEmbeddings, ComputeTableEmbeddings.Result> {

  /**
   * Choose the set of columns to compute embeddings, by default all columns are chosen. Embeddings
   * are computed only of columns of type string.
   *
   * @param columns a set of column names to be compute embeddings
   * @return this for method chaining
   */
  ComputeTableEmbeddings columns(String... columns);

  /**
   * Choose the table snapshot to compute embeddings, by default the current snapshot is used.
   *
   * @param snapshotId long ID of the snapshot for which embeddings need to be computed
   * @return this for method chaining
   */
  ComputeTableEmbeddings snapshot(long snapshotId);

  /**
   * Choose the model to use for computing embeddings, by default the in-process minilm model is
   * used.
   *
   * @param modelName model name used for computing embeddings.
   * @return this for method chaining
   */
  ComputeTableEmbeddings modelName(String modelName);

  /**
   * Choose the model inputs maps to use for model initialization, by default the empty map is used.
   *
   * @param modelInputs model inputs used for model initialization.
   * @return this for method chaining
   */
  ComputeTableEmbeddings modelInputs(Map<String, String> modelInputs);

  /** The result of table statistics collection. */
  interface Result {
    /** Returns statistics file or none if no statistics were collected. */
    StatisticsFile statisticsFile();
  }
}
