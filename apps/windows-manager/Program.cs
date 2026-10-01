using System.Diagnostics;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;

namespace NInfer.Manager;

internal static class Program
{
    internal static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web) { WriteIndented = true };

    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Length == 2 && args[0] == "--signal-stop" && int.TryParse(args[1], out var target))
            return NativeProcess.SignalStop(target) ? 0 : 1;
        ManagerPaths? paths = null;
        try
        {
            ApplicationConfiguration.Initialize();
            paths = ManagerPaths.Select(Value(args, "--root") ?? AppContext.BaseDirectory);
            using var instance = new Mutex(true, @"Local\NInferManager-" + paths.InstallationId, out var first);
            var sessionPath = Path.Combine(paths.RuntimeRoot, "session.json");
            if (!first)
            {
                if (!args.Contains("--autostart"))
                {
                    try
                    {
                        using var session = JsonDocument.Parse(File.ReadAllText(sessionPath));
                        using var owner = Process.GetProcessById(session.RootElement.GetProperty("pid").GetInt32());
                        if (!owner.HasExited && owner.MainModule?.FileName == Environment.ProcessPath)
                            OpenBrowser(session.RootElement.GetProperty("browserUrl").GetString()!);
                    }
                    catch { /* An already starting instance owns the launch. */ }
                }
                return 0;
            }
            try { return Run(paths, sessionPath, args); }
            finally
            {
                try { if (File.Exists(sessionPath)) File.Delete(sessionPath); }
                finally { instance.ReleaseMutex(); }
            }
        }
        catch (Exception ex)
        {
            var logPath = paths is null ? null : Path.Combine(paths.LogsRoot, "manager-errors.log");
            try { if (logPath is not null) File.AppendAllText(logPath, $"{DateTimeOffset.Now:o} {ex}\n"); }
            catch (Exception logError) when (logError is IOException or UnauthorizedAccessException) { logPath = null; }
            MessageBox.Show("NInfer 无法启动 / NInfer could not start\n\n" + ex.Message +
                (logPath is null ? "" : "\n\n" + logPath), "NInfer Manager", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }

    private static int Run(ManagerPaths paths, string sessionPath, string[] args)
    {
        var root = paths.PackageRoot;
        var store = new ConfigurationStore(paths);
        var controller = new EngineController(paths.DataRoot);
        var port = store.Settings.WebPort;
        var origin = $"http://127.0.0.1:{port}";
        var builder = WebApplication.CreateSlimBuilder(new WebApplicationOptions
        {
            Args = [], ContentRootPath = AppContext.BaseDirectory,
            WebRootPath = Path.Combine(AppContext.BaseDirectory, "wwwroot")
        });
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls(origin);
        var web = builder.Build();
        web.Use(async (context, next) =>
        {
            context.Response.Headers.CacheControl = "no-store";
            context.Response.Headers.XContentTypeOptions = "nosniff";
            context.Response.Headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'";
            await next(context);
        });
        using var tray = new TrayContext(store, controller, origin);
        ManagerApi.Map(web, store, controller, tray.RequestExit, () =>
        {
            StartupRegistration.SetEnabled(Environment.ProcessPath!, root, store.Settings.StartWithWindows);
            tray.RefreshSoon();
        });
        web.UseDefaultFiles();
        web.UseStaticFiles();
        web.MapFallbackToFile("index.html");
        try
        {
            web.StartAsync().GetAwaiter().GetResult();
            File.WriteAllText(sessionPath, JsonSerializer.Serialize(new
            { pid = Environment.ProcessId, baseUrl = origin, browserUrl = origin + "/",
                packageRoot = paths.PackageRoot, dataRoot = paths.DataRoot }, Json));
            bool skipAuto = args.Contains("--no-autostart");
            if (!skipAuto)
            {
                if (store.Settings.AutoStartModel) tray.StartDefault();
            }
            if (args.Contains("--open")) tray.OpenPage("/");
            Application.Run(tray);
            return 0;
        }
        finally
        {
            controller.DisposeAsync().AsTask().GetAwaiter().GetResult();
            web.StopAsync(TimeSpan.FromSeconds(5)).GetAwaiter().GetResult();
            web.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    internal static string? Value(string[] args, string name)
    {
        int index = Array.IndexOf(args, name);
        if (index < 0) return null;
        if (index + 1 >= args.Length) throw new ArgumentException($"Missing value for {name}");
        return args[index + 1];
    }

    internal static void OpenBrowser(string url) => Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
}
