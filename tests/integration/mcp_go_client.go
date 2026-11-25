// Exercise the synthetic camera host with the independent Go MCP SDK.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image/jpeg"
	"os"
	"os/exec"
	"reflect"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func probe(python, fixture string) (err error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, python, fixture, "granted")
	command.Stderr = os.Stderr
	client := mcp.NewClient(&mcp.Implementation{Name: "imx-camera-probe", Version: "1"}, nil)
	session, err := client.Connect(ctx, &mcp.CommandTransport{Command: command}, nil)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, session.Close()) }()

	if discovery := session.InitializeResult(); discovery.ProtocolVersion != "2026-07-28" ||
		discovery.ServerInfo == nil || discovery.ServerInfo.Name != "imx-camera-toolkit" {
		return fmt.Errorf("unexpected server discovery: %+v", discovery)
	}
	list, err := session.ListTools(ctx, nil)
	if err != nil {
		return err
	}
	names := []string{"camera_status", "camera_capabilities", "camera_snapshot"}
	if len(list.Tools) != len(names) {
		return fmt.Errorf("expected %d tools, got %d", len(names), len(list.Tools))
	}
	for index, name := range names {
		if list.Tools[index].Name != name || list.Tools[index].OutputSchema == nil {
			return fmt.Errorf("unexpected tool at index %d: %+v", index, list.Tools[index])
		}
		result, err := session.CallTool(ctx, &mcp.CallToolParams{Name: name})
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		if result.IsError || result.StructuredContent == nil {
			return fmt.Errorf("%s did not return a successful structured result", name)
		}
		if name == "camera_snapshot" {
			if err := checkJPEG(result); err != nil {
				return err
			}
		}
	}
	return nil
}

func checkJPEG(result *mcp.CallToolResult) error {
	if len(result.Content) != 2 {
		return fmt.Errorf("expected two snapshot content blocks, got %d", len(result.Content))
	}
	text, hasText := result.Content[0].(*mcp.TextContent)
	image, hasImage := result.Content[1].(*mcp.ImageContent)
	if !hasText || !hasImage || image.MIMEType != "image/jpeg" {
		return errors.New("snapshot did not contain JSON text and a native JPEG")
	}
	var metadata map[string]any
	if err := json.Unmarshal([]byte(text.Text), &metadata); err != nil {
		return err
	}
	encoded, err := json.Marshal(result.StructuredContent)
	if err != nil {
		return err
	}
	var structured map[string]any
	if err := json.Unmarshal(encoded, &structured); err != nil {
		return err
	}
	if !reflect.DeepEqual(metadata, structured) {
		return errors.New("structured metadata differs from the text block")
	}
	decoded, err := jpeg.Decode(bytes.NewReader(image.Data))
	if err != nil {
		return err
	}
	if decoded.Bounds().Dx() != 1 || decoded.Bounds().Dy() != 1 ||
		metadata["width"] != 1.0 || metadata["height"] != 1.0 ||
		metadata["frame_sequence"] != 3.0 || metadata["age_ms"] != 25.0 ||
		metadata["mime_type"] != image.MIMEType || metadata["jpeg_bytes"] != float64(len(image.Data)) {
		return fmt.Errorf("unexpected JPEG dimensions or metadata: %v", metadata)
	}
	return nil
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: mcp_go_client /path/to/python /path/to/test_mcp_camera_tools.py")
		os.Exit(2)
	}
	if err := probe(os.Args[1], os.Args[2]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("MCP 2026-07-28 discovery, three camera tools, and matching metadata for a decoded 1x1 JPEG")
}
