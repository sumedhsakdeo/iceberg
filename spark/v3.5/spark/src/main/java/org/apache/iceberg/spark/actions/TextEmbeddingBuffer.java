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
package org.apache.iceberg.spark.actions;

import com.google.gson.GsonBuilder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class TextEmbeddingBuffer {

  private final List<TextEmbedding> textEmbeddings;

  public TextEmbeddingBuffer(List<TextEmbedding> textEmbeddings) {
    this.textEmbeddings = textEmbeddings;
  }

  public static TextEmbeddingBuffer deserialize(byte[] bytes) {
    String jsonString = new String(bytes, StandardCharsets.UTF_8);
    GsonBuilder gsonBuilder = new GsonBuilder();
    gsonBuilder.registerTypeAdapter(
        TextEmbeddingBuffer.class, new TextEmbeddingBufferTypeAdapter<TextEmbeddingBuffer>());
    return gsonBuilder.create().fromJson(jsonString, TextEmbeddingBuffer.class);
  }

  public List<TextEmbedding> getTextEmbeddings() {
    return this.textEmbeddings;
  }

  public TextEmbeddingBuffer merge(TextEmbeddingBuffer other) {
    return new TextEmbeddingBuffer(
        Stream.concat(this.getTextEmbeddings().stream(), other.getTextEmbeddings().stream())
            .collect(Collectors.toList()));
  }

  public byte[] serialize() {
    GsonBuilder gsonBuilder = new GsonBuilder();
    gsonBuilder.registerTypeAdapter(
        TextEmbeddingBuffer.class, new TextEmbeddingBufferTypeAdapter<TextEmbeddingBuffer>());
    return gsonBuilder.create().toJson(this).getBytes(StandardCharsets.UTF_8);
  }

  @Override
  public String toString() {
    GsonBuilder gsonBuilder = new GsonBuilder();
    gsonBuilder.registerTypeAdapter(
        TextEmbeddingBuffer.class, new TextEmbeddingBufferTypeAdapter<TextEmbeddingBuffer>());
    return gsonBuilder.create().toJson(this);
  }

  public static TextEmbeddingBuffer fromString(String json) {
    GsonBuilder gsonBuilder = new GsonBuilder();
    gsonBuilder.registerTypeAdapter(
        TextEmbeddingBuffer.class, new TextEmbeddingBufferTypeAdapter<TextEmbeddingBuffer>());
    return gsonBuilder.create().fromJson(json, TextEmbeddingBuffer.class);
  }
}
