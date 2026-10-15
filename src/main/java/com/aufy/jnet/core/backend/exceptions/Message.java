package com.aufy.jnet.core.backend.exceptions;

public class Message {
  private static String methodString(String operationName) {
    return (operationName == null || operationName.isBlank()) ? "" : "for '" + operationName + "'";
  }

  /**
   * Stops all processes and exits. It follows the format:
   * 
   * > (with an explicit operation) [responsibility ERROR] messege for operationName: cause
   * 
   * > (with operation as Null) [responsibility ERROR] messege: cause
   * 
   * @param responsibility The kind of error. Always in CAPS.
   * @param messege The message to print.
   * @param operationName The name of the operation (method). It can also be Null so that it won't be printed.
   * @param cause The cause of the error.
   */
  public static void crash(String responsibility, String messege, String operationName, String cause) {
    System.out.println("[" + responsibility.toUpperCase() + " ERROR] " + messege + " " + methodString(operationName) + ": " + cause);
    System.exit(0);
  }

  /**
   * Prints a warning message.
   * 
   * @param messege The message to print.
   * @param operationName The name of the operation (method).
   * @param cause The cause of the error.
   */
  public static void warning(String messege, String operationName, String cause) {
    System.out.println("[WARNING] " + messege + " " + methodString(operationName) + ": " + cause);
  }
}
