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

package org.apache.iceberg.spark.actions;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import java.time.Duration;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class EmbeddingModelBuilder {

    // Models that run in the JVM.
    public static final String ALL_MINI_LM_L6V2 = "allminilml6v2";

    // Models that have dependencies to remote services.
    public static final String OLLAMA_LLAMA_31 = "ollama/llama3.1";

    // Model Defaults
    // Ollama model input keys
    public static final String OLLAMA_BASE_URL = "baseUrl";
    public static final String OLLAMA_TIMEOUT_SECS = "timeoutSecs";
    public static final String OLLAMA_LOG_REQUEST = "logRequest";
    public static final String OLLAMA_LOG_RESPONSE = "logResponse";
    private static final String OLLAMA_MAX_RETRIES = "maxRetries";
    // Ollama model input default values
    private static final String OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/";
    private static final String OLLAMA_DEFAULT_TIMEOUT_SECS = "10";
    private static final String OLLAMA_DEFAULT_MAX_RETRIES = "3";
    private static final String OLLAMA_DEFAULT_LOG_REQUEST = "false";
    private static final String OLLAMA_DEFAULT_LOG_RESPONSE = "false";

    private String modelName;
    private Map<String, String> modelInputs;

    public static EmbeddingModelBuilder builder() {
        return new EmbeddingModelBuilder();
    }

    public EmbeddingModelBuilder modelName(String modelName) {
        this.modelName = modelName;
        return this;
    }

    public EmbeddingModelBuilder modelInputs(Map<String, String> modelInputs) {
        this.modelInputs = modelInputs;
        return this;
    }

    public EmbeddingModel build() {
        switch (this.modelName) {
            case ALL_MINI_LM_L6V2:
                return buildAllMiniLmL6V2EmbeddingModel();
            case OLLAMA_LLAMA_31:
                return buildOllamaEmbeddingModel(modelName, modelInputs);
            default:
                throw new UnsupportedOperationException("Model is not supported");
        }
    }

    private AllMiniLmL6V2EmbeddingModel buildAllMiniLmL6V2EmbeddingModel() {
        System.setProperty("ai.djl.offline", "true");
        System.setProperty("DJL_OFFLINE", "true");
        return new AllMiniLmL6V2EmbeddingModel();
    }

    private OllamaEmbeddingModel buildOllamaEmbeddingModel(String modelName, Map<String, String> modelInputs) {
        String baseUrl = modelInputs.getOrDefault(OLLAMA_BASE_URL,
                OLLAMA_DEFAULT_BASE_URL);
        String ollamaModelName = modelName.split("/")[1];
        Duration duration = Duration.ofSeconds(Long.parseLong(modelInputs.getOrDefault(OLLAMA_TIMEOUT_SECS, OLLAMA_DEFAULT_TIMEOUT_SECS)));
        int maxRetries = Integer.parseInt(modelInputs.getOrDefault(OLLAMA_MAX_RETRIES, OLLAMA_DEFAULT_MAX_RETRIES));
        boolean logRequest = Boolean.parseBoolean(modelInputs.getOrDefault(OLLAMA_LOG_REQUEST, OLLAMA_DEFAULT_LOG_REQUEST));
        boolean logResponse = Boolean.parseBoolean(modelInputs.getOrDefault(OLLAMA_LOG_RESPONSE, OLLAMA_DEFAULT_LOG_RESPONSE));
        Set<String> skippedKeys = Set.of(OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SECS, OLLAMA_MAX_RETRIES, OLLAMA_LOG_REQUEST, OLLAMA_LOG_RESPONSE);
        return new OllamaEmbeddingModel(
                baseUrl,
                ollamaModelName,
                duration,
                maxRetries,
                logRequest,
                logResponse,
                modelInputs.entrySet().stream()
                        .filter(entry -> !skippedKeys.contains(entry.getKey()))
                        .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));
    }

}
