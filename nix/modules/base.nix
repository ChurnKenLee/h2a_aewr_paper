{ pkgs, ... }:

{
  packages = with pkgs; [
    actionlint
    coreutils
    curl
    git
    gnugrep
    gnumake
    gnused
    jq
    zola
  ];

  scripts.agent-context = {
    description = "Inspect, generate, verify, or review coding-agent context";
    exec = ''
      exec python scripts/agent_docs.py "$@"
    '';
  };

  scripts.agent-docs-generate = {
    description = "Regenerate machine-readable context for the Zola documentation";
    exec = ''
      exec python scripts/agent_docs.py generate
    '';
  };

  scripts.agent-docs-check = {
    description = "Verify agent context, its tests, and the Zola site";
    exec = ''
      python scripts/agent_docs.py verify
      python scripts/test_agent_docs.py
      actionlint
      zola check --skip-external-links
      exec zola build
    '';
  };

  scripts.agent-docs-serve = {
    description = "Verify and serve the grounded documentation";
    exec = ''
      python scripts/agent_docs.py verify
      exec zola serve "$@"
    '';
  };
}
