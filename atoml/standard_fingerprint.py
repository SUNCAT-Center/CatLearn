<!DOCTYPE html>
<html class="" lang="en">
<head prefix="og: http://ogp.me/ns#">
<meta charset="utf-8">
<meta content="IE=edge" http-equiv="X-UA-Compatible">
<meta content="object" property="og:type">
<meta content="GitLab" property="og:site_name">
<meta content="atoml/standard_fingerprint.py · a66d036ae1a47e80bfa4f39601fbb89cff87e3cc · atoML / AtoML" property="og:title">
<meta content="Machine Learning using atomic-scale calculations." property="og:description">
<meta content="https://gitlab.com/assets/gitlab_logo-7ae504fe4f68fdebb3c2034e36621930cd36ea87924c11ff65dbcb8ed50dca58.png" property="og:image">
<meta content="https://gitlab.com/atoML/AtoML/blob/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc/atoml/standard_fingerprint.py" property="og:url">
<meta content="summary" property="twitter:card">
<meta content="atoml/standard_fingerprint.py · a66d036ae1a47e80bfa4f39601fbb89cff87e3cc · atoML / AtoML" property="twitter:title">
<meta content="Machine Learning using atomic-scale calculations." property="twitter:description">
<meta content="https://gitlab.com/assets/gitlab_logo-7ae504fe4f68fdebb3c2034e36621930cd36ea87924c11ff65dbcb8ed50dca58.png" property="twitter:image">

<title>atoml/standard_fingerprint.py · a66d036ae1a47e80bfa4f39601fbb89cff87e3cc · atoML / AtoML · GitLab</title>
<meta content="Machine Learning using atomic-scale calculations." name="description">
<link rel="shortcut icon" type="image/x-icon" href="/assets/favicon-075eba76312e8421991a0c1f89a89ee81678bcde72319dd3e8047e2a47cd3a42.ico" />
<link rel="stylesheet" media="all" href="/assets/application-a6dd150d84720bf9a3c0d83ce742846db842b2f38248e1dd91159801d5aa5f41.css" />
<link rel="stylesheet" media="print" href="/assets/print-9c3a1eb4a2f45c9f3d7dd4de03f14c2e6b921e757168b595d7f161bbc320fc05.css" />
<script src="/assets/application-8d6d7877bb93d20f2080ff81b36d47fe748b20b0b0f6cd659ca42fcb2ef3e2fd.js"></script>
<meta name="csrf-param" content="authenticity_token" />
<meta name="csrf-token" content="YfOHXcvv55FNjg+220MuV8qHsfDwac+jM+13FHcUPQkqRrCfOAk/miO12eV966JL/kRbDLeg0iGqhCLmo884fg==" />
<meta content="origin-when-cross-origin" name="referrer">
<meta content="width=device-width, initial-scale=1, maximum-scale=1" name="viewport">
<meta content="#474D57" name="theme-color">
<link rel="apple-touch-icon" type="image/x-icon" href="/assets/touch-icon-iphone-5a9cee0e8a51212e70b90c87c12f382c428870c0ff67d1eb034d884b78d2dae7.png" />
<link rel="apple-touch-icon" type="image/x-icon" href="/assets/touch-icon-ipad-a6eec6aeb9da138e507593b464fdac213047e49d3093fc30e90d9a995df83ba3.png" sizes="76x76" />
<link rel="apple-touch-icon" type="image/x-icon" href="/assets/touch-icon-iphone-retina-72e2aadf86513a56e050e7f0f2355deaa19cc17ed97bbe5147847f2748e5a3e3.png" sizes="120x120" />
<link rel="apple-touch-icon" type="image/x-icon" href="/assets/touch-icon-ipad-retina-8ebe416f5313483d9c1bc772b5bbe03ecad52a54eba443e5215a22caed2a16a2.png" sizes="152x152" />
<link color="rgb(226, 67, 41)" href="/assets/logo-d36b5212042cebc89b96df4bf6ac24e43db316143e89926c0db839ff694d2de4.svg" rel="mask-icon">
<meta content="/assets/msapplication-tile-1196ec67452f618d39cdd85e2e3a542f76574c071051ae7effbfde01710eb17d.png" name="msapplication-TileImage">
<meta content="#30353E" name="msapplication-TileColor">


<!-- Piwik -->
<script>
  var _paq = _paq || [];
  _paq.push(['trackPageView']);
  _paq.push(['enableLinkTracking']);
  (function() {
    var u="//piwik.gitlab.com/";
    _paq.push(['setTrackerUrl', u+'piwik.php']);
    _paq.push(['setSiteId', 1]);
    var d=document, g=d.createElement('script'), s=d.getElementsByTagName('script')[0];
    g.type='text/javascript'; g.async=true; g.defer=true; g.src=u+'piwik.js'; s.parentNode.insertBefore(g,s);
  })();
</script>
<noscript><p><img src="//piwik.gitlab.com/piwik.php?idsite=1" style="border:0;" alt="" /></p></noscript>
<!-- End Piwik Code -->


</head>

<body class="ui_charcoal" data-group="" data-page="projects:blob:show" data-project="AtoML">
<script>
//<![CDATA[
window.gon={};gon.api_version="v3";gon.default_avatar_url="https:\/\/gitlab.com\/assets\/no_avatar-849f9c04a3a0d0cea2424ae97b27447dc64a7dbfae83c036c45b403392f0e8ba.png";gon.max_file_size=10;gon.relative_url_root="";gon.shortcuts_path="\/help\/shortcuts";gon.user_color_scheme="white";gon.award_menu_url="\/emojis";gon.katex_css_url="\/assets\/katex-e46cafe9c3fa73920a7c2c063ee8bb0613e0cf85fd96a3aea25f8419c4bfcfba.css";gon.katex_js_url="\/assets\/katex-04bcf56379fcda0ee7c7a63f71d0fc15ffd2e014d017cd9d51fd6554dfccf40a.js";gon.current_user_id=256176;gon.current_username="mhangaard";
//]]>
</script>
<script>
  window.project_uploads_path = "/atoML/AtoML/uploads";
  window.preview_markdown_path = "/atoML/AtoML/preview_markdown";
</script>

<header class="navbar navbar-fixed-top navbar-gitlab with-horizontal-nav">
<a class="sr-only gl-accessibility" href="#content-body" tabindex="1">Skip to content</a>
<div class="container-fluid">
<div class="header-content">
<button aria-label="Toggle global navigation" class="side-nav-toggle" type="button">
<span class="sr-only">Toggle navigation</span>
<i class="fa fa-bars"></i>
</button>
<button class="navbar-toggle" type="button">
<span class="sr-only">Toggle navigation</span>
<i class="fa fa-ellipsis-v"></i>
</button>
<div class="navbar-collapse collapse">
<ul class="nav navbar-nav">
<li class="hidden-sm hidden-xs">
<div class="has-location-badge search search-form">
<form class="navbar-form" action="/search" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" /><div class="search-input-container">
<div class="location-badge">This project</div>
<div class="search-input-wrap">
<div class="dropdown" data-url="/search/autocomplete">
<input type="search" name="search" id="search" placeholder="Search" class="search-input dropdown-menu-toggle no-outline js-search-dashboard-options" spellcheck="false" tabindex="1" autocomplete="off" data-toggle="dropdown" data-issues-path="https://gitlab.com/dashboard/issues" data-mr-path="https://gitlab.com/dashboard/merge_requests" />
<div class="dropdown-menu dropdown-select">
<div class="dropdown-content"><ul>
<li>
<a class="is-focused dropdown-menu-empty-link">
Loading...
</a>
</li>
</ul>
</div><div class="dropdown-loading"><i class="fa fa-spinner fa-spin"></i></div>
</div>
<i class="search-icon"></i>
<i class="clear-icon js-clear-input"></i>
</div>
</div>
</div>
<input type="hidden" name="group_id" id="group_id" class="js-search-group-options" />
<input type="hidden" name="project_id" id="search_project_id" value="2160056" class="js-search-project-options" data-project-path="AtoML" data-name="AtoML" data-issues-path="/atoML/AtoML/issues" data-mr-path="/atoML/AtoML/merge_requests" />
<input type="hidden" name="search_code" id="search_code" value="true" />
<input type="hidden" name="repository_ref" id="repository_ref" value="a66d036ae1a47e80bfa4f39601fbb89cff87e3cc" />

<div class="search-autocomplete-opts hide" data-autocomplete-path="/search/autocomplete" data-autocomplete-project-id="2160056" data-autocomplete-project-ref="a66d036ae1a47e80bfa4f39601fbb89cff87e3cc"></div>
</form></div>

</li>
<li class="visible-sm visible-xs">
<a title="Search" aria-label="Search" data-toggle="tooltip" data-placement="bottom" data-container="body" href="/search"><i class="fa fa-search"></i>
</a></li>
<li>
<a title="Todos" aria-label="Todos" data-toggle="tooltip" data-placement="bottom" data-container="body" href="/dashboard/todos"><i class="fa fa-bell fa-fw"></i>
<span class="badge hidden todos-pending-count">
0
</span>
</a></li>
<li class="header-user dropdown">
<a class="header-user-dropdown-toggle" data-toggle="dropdown" href="/mhangaard"><img width="26" height="26" class="header-user-avatar" src="https://gitlab.com/uploads/user/avatar/256176/Martin_Hangaard-Hansen.jpg" alt="Martin hangaard hansen" />
<i class="fa fa-caret-down"></i>
</a><div class="dropdown-menu-nav dropdown-menu-align-right">
<ul>
<li>
<a class="profile-link" aria-label="Profile" data-user="mhangaard" href="/mhangaard">Profile</a>
</li>
<li>
<a aria-label="Settings" href="/profile">Settings</a>
</li>
<li>
<a aria-label="Help" href="/help">Help</a>
</li>
<li class="divider"></li>
<li>
<a class="sign-out-link" aria-label="Sign out" rel="nofollow" data-method="delete" href="/users/sign_out">Sign out</a>
</li>
</ul>
</div>
</li>
</ul>
</div>
<h1 class="title"><a href="/atoML">atoML</a> / <a class="project-item-select-holder" href="/atoML/AtoML">AtoML</a><button name="button" type="button" class="dropdown-toggle-caret js-projects-dropdown-toggle" aria-label="Toggle switch project dropdown" data-target=".js-dropdown-menu-projects" data-toggle="dropdown" data-order-by="last_activity_at"><i class="fa fa-chevron-down"></i></button></h1>
<div class="header-logo">
<a class="home" title="Dashboard" id="logo" href="/"><svg width="36" height="36" class="tanuki-logo">
  <path class="tanuki-shape tanuki-left-ear" fill="#e24329" d="M2 14l9.38 9v-9l-4-12.28c-.205-.632-1.176-.632-1.38 0z"/>
  <path class="tanuki-shape tanuki-right-ear" fill="#e24329" d="M34 14l-9.38 9v-9l4-12.28c.205-.632 1.176-.632 1.38 0z"/>
  <path class="tanuki-shape tanuki-nose" fill="#e24329" d="M18,34.38 3,14 33,14 Z"/>
  <path class="tanuki-shape tanuki-left-eye" fill="#fc6d26" d="M18,34.38 11.38,14 2,14 6,25Z"/>
  <path class="tanuki-shape tanuki-right-eye" fill="#fc6d26" d="M18,34.38 24.62,14 34,14 30,25Z"/>
  <path class="tanuki-shape tanuki-left-cheek" fill="#fca326" d="M2 14L.1 20.16c-.18.565 0 1.2.5 1.56l17.42 12.66z"/>
  <path class="tanuki-shape tanuki-right-cheek" fill="#fca326" d="M34 14l1.9 6.16c.18.565 0 1.2-.5 1.56L18 34.38z"/>
</svg>

</a></div>
<div class="js-dropdown-menu-projects">
<div class="dropdown-menu dropdown-select dropdown-menu-projects">
<div class="dropdown-title"><span>Go to a project</span><button class="dropdown-title-button dropdown-menu-close" aria-label="Close" type="button"><i class="fa fa-times dropdown-menu-close-icon"></i></button></div>
<div class="dropdown-input"><input type="search" id="" class="dropdown-input-field" placeholder="Search your projects" autocomplete="off" /><i class="fa fa-search dropdown-input-search"></i><i role="button" class="fa fa-times dropdown-input-clear js-dropdown-input-clear"></i></div>
<div class="dropdown-content"></div>
<div class="dropdown-loading"><i class="fa fa-spinner fa-spin"></i></div>
</div>
</div>

</div>
</div>
</header>

<script>
  var findFileURL = "/atoML/AtoML/find_file/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc";
</script>

<div class="page-with-sidebar">
<div class="sidebar-wrapper nicescroll">
<div class="sidebar-action-buttons">
<div class="nav-header-btn toggle-nav-collapse" title="Open/Close">
<span class="sr-only">Toggle navigation</span>
<i class="fa fa-bars"></i>
</div>
<div class="nav-header-btn pin-nav-btn has-tooltip  js-nav-pin" data-container="body" data-placement="right" title="Pin Navigation">
<span class="sr-only">Toggle navigation pinning</span>
<i class="fa fa-fw fa-thumb-tack"></i>
</div>
</div>
<div class="nav-sidebar">
<ul class="nav">
<li class="active home"><a title="Projects" class="dashboard-shortcuts-projects" href="/dashboard/projects"><span>
Projects
</span>
</a></li><li class=""><a class="dashboard-shortcuts-activity" title="Activity" href="/dashboard/activity"><span>
Activity
</span>
</a></li><li class=""><a title="Groups" href="/dashboard/groups"><span>
Groups
</span>
</a></li><li class=""><a title="Milestones" href="/dashboard/milestones"><span>
Milestones
</span>
</a></li><li class=""><a title="Issues" class="dashboard-shortcuts-issues" href="/dashboard/issues?assignee_id=256176"><span>
Issues
<span class="count">0</span>
</span>
</a></li><li class=""><a title="Merge Requests" class="dashboard-shortcuts-merge_requests" href="/dashboard/merge_requests?assignee_id=256176"><span>
Merge Requests
<span class="count">0</span>
</span>
</a></li><li class=""><a title="Snippets" href="/dashboard/snippets"><span>
Snippets
</span>
</a></li><a title="About GitLab EE" class="about-gitlab" href="/help"><span>
About GitLab EE
</span>
</a></ul>
</div>

</div>
<div class="layout-nav">
<div class="container-fluid">
<div class="controls">
<div class="dropdown project-settings-dropdown">
<a class="dropdown-new btn btn-default" data-toggle="dropdown" href="#" id="project-settings-button">
<i class="fa fa-cog"></i>
<i class="fa fa-caret-down"></i>
</a>
<ul class="dropdown-menu dropdown-menu-align-right">
<li class=""><a title="Members" class="team-tab tab" href="/atoML/AtoML/settings/members"><span>
Members
</span>
</a></li><li class=""><a title="Deploy Keys" href="/atoML/AtoML/deploy_keys"><span>
Deploy Keys
</span>
</a></li><li class=""><a title="Integrations" href="/atoML/AtoML/settings/integrations"><span>
Integrations
</span>
</a></li><li class=""><a title="Protected Branches" href="/atoML/AtoML/protected_branches"><span>
Protected Branches
</span>
</a></li><li class=""><a title="Runners" href="/atoML/AtoML/runners"><span>
Runners
</span>
</a></li><li class=""><a title="Variables" href="/atoML/AtoML/variables"><span>
Variables
</span>
</a></li><li class=""><a title="Triggers" href="/atoML/AtoML/triggers"><span>
Triggers
</span>
</a></li><li class=""><a title="CI/CD Pipelines" href="/atoML/AtoML/pipelines/settings"><span>
CI/CD Pipelines
</span>
</a></li><li class=""><a title="Push Rules" href="/atoML/AtoML/push_rules"><span>
Push Rules
</span>
</a></li><li class=""><a title="Mirror Repository" data-placement="right" href="/atoML/AtoML/mirror"><span>
Mirror Repository
</span>
</a></li><li class=""><a title="Pages" data-placement="right" href="/atoML/AtoML/pages"><span>
Pages
</span>
</a></li><li class=""><a title="Audit Events" href="/atoML/AtoML/audit_events"><span>
Audit Events
</span>
</a></li>
<li class="divider"></li>
<li>
<a href="/atoML/AtoML/edit">Edit Project
</a></li>
</ul>
</div>
</div>
<div class="nav-control scrolling-tabs-container">
<div class="fade-left">
<i class="fa fa-angle-left"></i>
</div>
<div class="fade-right">
<i class="fa fa-angle-right"></i>
</div>
<ul class="nav-links scrolling-tabs">
<li class="home"><a title="Project" class="shortcuts-project" href="/atoML/AtoML"><span>
Project
</span>
</a></li><li class=""><a title="Activity" class="shortcuts-project-activity" href="/atoML/AtoML/activity"><span>
Activity
</span>
</a></li><li class="active"><a title="Repository" class="shortcuts-tree" href="/atoML/AtoML/tree/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc"><span>
Repository
</span>
</a></li><li class=""><a title="Pipelines" class="shortcuts-pipelines" href="/atoML/AtoML/pipelines"><span>
Pipelines
</span>
</a></li><li class=""><a title="Container Registry" class="shortcuts-container-registry" href="/atoML/AtoML/container_registry"><span>
Registry
</span>
</a></li><li class=""><a title="Graphs" class="shortcuts-graphs" href="/atoML/AtoML/graphs/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc"><span>
Graphs
</span>
</a></li><li class=""><a title="Issues" class="shortcuts-issues" href="/atoML/AtoML/issues"><span>
Issues
<span class="badge count issue_counter">0</span>
</span>
</a></li><li class=""><a title="Merge Requests" class="shortcuts-merge_requests" href="/atoML/AtoML/merge_requests"><span>
Merge Requests
<span class="badge count merge_counter">0</span>
</span>
</a></li><li class=""><a title="Wiki" class="shortcuts-wiki" href="/atoML/AtoML/wikis/home"><span>
Wiki
</span>
</a></li><li class="hidden">
<a title="Network" class="shortcuts-network" href="/atoML/AtoML/network/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc">Network
</a></li>
<li class="hidden">
<a class="shortcuts-new-issue" href="/atoML/AtoML/issues/new">Create a new issue
</a></li>
<li class="hidden">
<a title="Builds" class="shortcuts-builds" href="/atoML/AtoML/builds">Builds
</a></li>
<li class="hidden">
<a title="Commits" class="shortcuts-commits" href="/atoML/AtoML/commits/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc">Commits
</a></li>
<li class="hidden">
<a title="Issue Boards" class="shortcuts-issue-boards" href="/atoML/AtoML/boards">Issue Boards</a>
</li>
</ul>
</div>

</div>
</div>
<div class="content-wrapper page-with-layout-nav">
<div class="scrolling-tabs-container sub-nav-scroll">
<div class="fade-left">
<i class="fa fa-angle-left"></i>
</div>
<div class="fade-right">
<i class="fa fa-angle-right"></i>
</div>

<div class="nav-links sub-nav scrolling-tabs">
<ul class="container-fluid container-limited">
<li class="active"><a href="/atoML/AtoML/tree/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc">Files
</a></li><li class=""><a href="/atoML/AtoML/commits/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc">Commits
</a></li><li class=""><a href="/atoML/AtoML/network/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc">Network
</a></li><li class=""><a href="/atoML/AtoML/compare?from=master&amp;to=a66d036ae1a47e80bfa4f39601fbb89cff87e3cc">Compare
</a></li><li class=""><a href="/atoML/AtoML/branches">Branches
</a></li><li class=""><a href="/atoML/AtoML/tags">Tags
</a></li><li class=""><a href="/atoML/AtoML/path_locks">Locked Files
</a></li></ul>
</div>
</div>

<div class="alert-wrapper">


<div class="flash-container flash-container-page">
</div>


</div>
<div class=" ">
<div class="content" id="content-body">

<div class="container-fluid container-limited">

<div class="tree-holder" id="tree-holder">
<div class="nav-block">
<div class="tree-ref-holder">
<form class="project-refs-form" action="/atoML/AtoML/refs/switch" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="destination" id="destination" value="blob" />
<input type="hidden" name="path" id="path" value="atoml/standard_fingerprint.py" />
<div class="dropdown">
<button class="dropdown-menu-toggle js-project-refs-dropdown" type="button" data-toggle="dropdown" data-selected="a66d036ae1a47e80bfa4f39601fbb89cff87e3cc" data-ref="a66d036ae1a47e80bfa4f39601fbb89cff87e3cc" data-refs-url="/atoML/AtoML/refs" data-field-name="ref" data-submit-form-on-click="true"><span class="dropdown-toggle-text ">a66d036ae1a47e80bfa4f39601fbb89cff87e3cc</span><i class="fa fa-chevron-down"></i></button>
<div class="dropdown-menu dropdown-menu-selectable">
<div class="dropdown-title"><span>Switch branch/tag</span><button class="dropdown-title-button dropdown-menu-close" aria-label="Close" type="button"><i class="fa fa-times dropdown-menu-close-icon"></i></button></div>
<div class="dropdown-input"><input type="search" id="" class="dropdown-input-field" placeholder="Search branches and tags" autocomplete="off" /><i class="fa fa-search dropdown-input-search"></i><i role="button" class="fa fa-times dropdown-input-clear js-dropdown-input-clear"></i></div>
<div class="dropdown-content"></div>
<div class="dropdown-loading"><i class="fa fa-spinner fa-spin"></i></div>
</div>
</div>
</form>
</div>
<ul class="breadcrumb repo-breadcrumb">
<li>
<a href="/atoML/AtoML/tree/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc">AtoML
</a></li>
<li>
<a href="/atoML/AtoML/tree/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc/atoml">atoml</a>

</li>
<li>
<a href="/atoML/AtoML/blob/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc/atoml/standard_fingerprint.py"><strong>
standard_fingerprint.py
</strong>
</a>
</li>
</ul>
</div>
<ul class="blob-commit-info table-list hidden-xs">
<li class="commit table-list-row js-toggle-container" id="commit-a66d036a">
<div class="table-list-cell avatar-cell hidden-xs">
<a href="/jennings.p.c"><img class="avatar has-tooltip s36 hidden-xs" alt="Paul C. Jennings&#39;s avatar" title="Paul C. Jennings" data-container="body" src="https://gitlab.com/uploads/user/avatar/269557/profile.jpg" /></a>
</div>
<div class="table-list-cell commit-content">
<a class="commit-row-message item-title" href="/atoML/AtoML/commit/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc">Fix small bug with distance_fpv.</a>
<span class="commit-row-message visible-xs-inline">
&middot;
a66d036a
</span>
<div class="commiter">
<a class="commit-author-link has-tooltip" title="jennings.p.c@gmail.com" href="/jennings.p.c">Paul C. Jennings</a>
committed
<time class="js-timeago" title="Feb 8, 2017 6:09pm" datetime="2017-02-08T18:09:09Z" data-toggle="tooltip" data-placement="top" data-container="body">2017-02-08 10:09:09 -0800</time>
</div>
</div>
<div class="table-list-cell commit-actions hidden-xs">
<button class="btn btn-clipboard btn-transparent" data-toggle="tooltip" data-placement="bottom" data-container="body" data-clipboard-text="a66d036ae1a47e80bfa4f39601fbb89cff87e3cc" type="button" title="Copy to clipboard"><i class="fa fa-clipboard"></i></button>
<a class="commit-short-id btn btn-transparent" href="/atoML/AtoML/commit/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc">a66d036a</a>

</div>
</li>

</ul>
<div class="blob-content-holder" id="blob-content-holder">
<article class="file-holder">
<div class="file-title">
<i class="fa fa-file-text-o fa-fw"></i>
<strong>
standard_fingerprint.py
</strong>
<small>
4.23 KB
</small>
<div class="file-actions hidden-xs">
<div class="btn-group tree-btn-group">
<a class="btn btn-sm" target="_blank" href="/atoML/AtoML/raw/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc/atoml/standard_fingerprint.py">Raw</a>
<a class="btn btn-sm" href="/atoML/AtoML/blame/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc/atoml/standard_fingerprint.py">Blame</a>
<a class="btn btn-sm" href="/atoML/AtoML/commits/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc/atoml/standard_fingerprint.py">History</a>
<a class="btn btn-sm" href="/atoML/AtoML/blob/a66d036ae1a47e80bfa4f39601fbb89cff87e3cc/atoml/standard_fingerprint.py">Permalink</a>
</div>
<div class="btn-group" role="group">
<a class="btn btn-sm path-lock has-tooltip" data-state="lock" data-toggle="tooltip" title="" href="#">Lock</a>
<button name="button" type="submit" class="btn disabled has-tooltip btn-file-option" title="You can only edit files when you are on a branch" data-container="body">Edit</button>
<button name="button" type="submit" class="btn btn-default disabled has-tooltip" title="You can only replace files when you are on a branch" data-container="body">Replace</button>
<button name="button" type="submit" class="btn btn-remove disabled has-tooltip" title="You can only delete files when you are on a branch" data-container="body">Delete</button>
</div>
<script>
  PathLocks.init(
    '/atoML/AtoML/path_locks/toggle',
    'atoml/standard_fingerprint.py'
  );
</script>

</div>
</div>
<div class="file-content code js-syntax-highlight">
<div class="line-numbers">
<a class="diff-line-num" data-line-number="1" href="#L1" id="L1">
<i class="fa fa-link"></i>
1
</a>
<a class="diff-line-num" data-line-number="2" href="#L2" id="L2">
<i class="fa fa-link"></i>
2
</a>
<a class="diff-line-num" data-line-number="3" href="#L3" id="L3">
<i class="fa fa-link"></i>
3
</a>
<a class="diff-line-num" data-line-number="4" href="#L4" id="L4">
<i class="fa fa-link"></i>
4
</a>
<a class="diff-line-num" data-line-number="5" href="#L5" id="L5">
<i class="fa fa-link"></i>
5
</a>
<a class="diff-line-num" data-line-number="6" href="#L6" id="L6">
<i class="fa fa-link"></i>
6
</a>
<a class="diff-line-num" data-line-number="7" href="#L7" id="L7">
<i class="fa fa-link"></i>
7
</a>
<a class="diff-line-num" data-line-number="8" href="#L8" id="L8">
<i class="fa fa-link"></i>
8
</a>
<a class="diff-line-num" data-line-number="9" href="#L9" id="L9">
<i class="fa fa-link"></i>
9
</a>
<a class="diff-line-num" data-line-number="10" href="#L10" id="L10">
<i class="fa fa-link"></i>
10
</a>
<a class="diff-line-num" data-line-number="11" href="#L11" id="L11">
<i class="fa fa-link"></i>
11
</a>
<a class="diff-line-num" data-line-number="12" href="#L12" id="L12">
<i class="fa fa-link"></i>
12
</a>
<a class="diff-line-num" data-line-number="13" href="#L13" id="L13">
<i class="fa fa-link"></i>
13
</a>
<a class="diff-line-num" data-line-number="14" href="#L14" id="L14">
<i class="fa fa-link"></i>
14
</a>
<a class="diff-line-num" data-line-number="15" href="#L15" id="L15">
<i class="fa fa-link"></i>
15
</a>
<a class="diff-line-num" data-line-number="16" href="#L16" id="L16">
<i class="fa fa-link"></i>
16
</a>
<a class="diff-line-num" data-line-number="17" href="#L17" id="L17">
<i class="fa fa-link"></i>
17
</a>
<a class="diff-line-num" data-line-number="18" href="#L18" id="L18">
<i class="fa fa-link"></i>
18
</a>
<a class="diff-line-num" data-line-number="19" href="#L19" id="L19">
<i class="fa fa-link"></i>
19
</a>
<a class="diff-line-num" data-line-number="20" href="#L20" id="L20">
<i class="fa fa-link"></i>
20
</a>
<a class="diff-line-num" data-line-number="21" href="#L21" id="L21">
<i class="fa fa-link"></i>
21
</a>
<a class="diff-line-num" data-line-number="22" href="#L22" id="L22">
<i class="fa fa-link"></i>
22
</a>
<a class="diff-line-num" data-line-number="23" href="#L23" id="L23">
<i class="fa fa-link"></i>
23
</a>
<a class="diff-line-num" data-line-number="24" href="#L24" id="L24">
<i class="fa fa-link"></i>
24
</a>
<a class="diff-line-num" data-line-number="25" href="#L25" id="L25">
<i class="fa fa-link"></i>
25
</a>
<a class="diff-line-num" data-line-number="26" href="#L26" id="L26">
<i class="fa fa-link"></i>
26
</a>
<a class="diff-line-num" data-line-number="27" href="#L27" id="L27">
<i class="fa fa-link"></i>
27
</a>
<a class="diff-line-num" data-line-number="28" href="#L28" id="L28">
<i class="fa fa-link"></i>
28
</a>
<a class="diff-line-num" data-line-number="29" href="#L29" id="L29">
<i class="fa fa-link"></i>
29
</a>
<a class="diff-line-num" data-line-number="30" href="#L30" id="L30">
<i class="fa fa-link"></i>
30
</a>
<a class="diff-line-num" data-line-number="31" href="#L31" id="L31">
<i class="fa fa-link"></i>
31
</a>
<a class="diff-line-num" data-line-number="32" href="#L32" id="L32">
<i class="fa fa-link"></i>
32
</a>
<a class="diff-line-num" data-line-number="33" href="#L33" id="L33">
<i class="fa fa-link"></i>
33
</a>
<a class="diff-line-num" data-line-number="34" href="#L34" id="L34">
<i class="fa fa-link"></i>
34
</a>
<a class="diff-line-num" data-line-number="35" href="#L35" id="L35">
<i class="fa fa-link"></i>
35
</a>
<a class="diff-line-num" data-line-number="36" href="#L36" id="L36">
<i class="fa fa-link"></i>
36
</a>
<a class="diff-line-num" data-line-number="37" href="#L37" id="L37">
<i class="fa fa-link"></i>
37
</a>
<a class="diff-line-num" data-line-number="38" href="#L38" id="L38">
<i class="fa fa-link"></i>
38
</a>
<a class="diff-line-num" data-line-number="39" href="#L39" id="L39">
<i class="fa fa-link"></i>
39
</a>
<a class="diff-line-num" data-line-number="40" href="#L40" id="L40">
<i class="fa fa-link"></i>
40
</a>
<a class="diff-line-num" data-line-number="41" href="#L41" id="L41">
<i class="fa fa-link"></i>
41
</a>
<a class="diff-line-num" data-line-number="42" href="#L42" id="L42">
<i class="fa fa-link"></i>
42
</a>
<a class="diff-line-num" data-line-number="43" href="#L43" id="L43">
<i class="fa fa-link"></i>
43
</a>
<a class="diff-line-num" data-line-number="44" href="#L44" id="L44">
<i class="fa fa-link"></i>
44
</a>
<a class="diff-line-num" data-line-number="45" href="#L45" id="L45">
<i class="fa fa-link"></i>
45
</a>
<a class="diff-line-num" data-line-number="46" href="#L46" id="L46">
<i class="fa fa-link"></i>
46
</a>
<a class="diff-line-num" data-line-number="47" href="#L47" id="L47">
<i class="fa fa-link"></i>
47
</a>
<a class="diff-line-num" data-line-number="48" href="#L48" id="L48">
<i class="fa fa-link"></i>
48
</a>
<a class="diff-line-num" data-line-number="49" href="#L49" id="L49">
<i class="fa fa-link"></i>
49
</a>
<a class="diff-line-num" data-line-number="50" href="#L50" id="L50">
<i class="fa fa-link"></i>
50
</a>
<a class="diff-line-num" data-line-number="51" href="#L51" id="L51">
<i class="fa fa-link"></i>
51
</a>
<a class="diff-line-num" data-line-number="52" href="#L52" id="L52">
<i class="fa fa-link"></i>
52
</a>
<a class="diff-line-num" data-line-number="53" href="#L53" id="L53">
<i class="fa fa-link"></i>
53
</a>
<a class="diff-line-num" data-line-number="54" href="#L54" id="L54">
<i class="fa fa-link"></i>
54
</a>
<a class="diff-line-num" data-line-number="55" href="#L55" id="L55">
<i class="fa fa-link"></i>
55
</a>
<a class="diff-line-num" data-line-number="56" href="#L56" id="L56">
<i class="fa fa-link"></i>
56
</a>
<a class="diff-line-num" data-line-number="57" href="#L57" id="L57">
<i class="fa fa-link"></i>
57
</a>
<a class="diff-line-num" data-line-number="58" href="#L58" id="L58">
<i class="fa fa-link"></i>
58
</a>
<a class="diff-line-num" data-line-number="59" href="#L59" id="L59">
<i class="fa fa-link"></i>
59
</a>
<a class="diff-line-num" data-line-number="60" href="#L60" id="L60">
<i class="fa fa-link"></i>
60
</a>
<a class="diff-line-num" data-line-number="61" href="#L61" id="L61">
<i class="fa fa-link"></i>
61
</a>
<a class="diff-line-num" data-line-number="62" href="#L62" id="L62">
<i class="fa fa-link"></i>
62
</a>
<a class="diff-line-num" data-line-number="63" href="#L63" id="L63">
<i class="fa fa-link"></i>
63
</a>
<a class="diff-line-num" data-line-number="64" href="#L64" id="L64">
<i class="fa fa-link"></i>
64
</a>
<a class="diff-line-num" data-line-number="65" href="#L65" id="L65">
<i class="fa fa-link"></i>
65
</a>
<a class="diff-line-num" data-line-number="66" href="#L66" id="L66">
<i class="fa fa-link"></i>
66
</a>
<a class="diff-line-num" data-line-number="67" href="#L67" id="L67">
<i class="fa fa-link"></i>
67
</a>
<a class="diff-line-num" data-line-number="68" href="#L68" id="L68">
<i class="fa fa-link"></i>
68
</a>
<a class="diff-line-num" data-line-number="69" href="#L69" id="L69">
<i class="fa fa-link"></i>
69
</a>
<a class="diff-line-num" data-line-number="70" href="#L70" id="L70">
<i class="fa fa-link"></i>
70
</a>
<a class="diff-line-num" data-line-number="71" href="#L71" id="L71">
<i class="fa fa-link"></i>
71
</a>
<a class="diff-line-num" data-line-number="72" href="#L72" id="L72">
<i class="fa fa-link"></i>
72
</a>
<a class="diff-line-num" data-line-number="73" href="#L73" id="L73">
<i class="fa fa-link"></i>
73
</a>
<a class="diff-line-num" data-line-number="74" href="#L74" id="L74">
<i class="fa fa-link"></i>
74
</a>
<a class="diff-line-num" data-line-number="75" href="#L75" id="L75">
<i class="fa fa-link"></i>
75
</a>
<a class="diff-line-num" data-line-number="76" href="#L76" id="L76">
<i class="fa fa-link"></i>
76
</a>
<a class="diff-line-num" data-line-number="77" href="#L77" id="L77">
<i class="fa fa-link"></i>
77
</a>
<a class="diff-line-num" data-line-number="78" href="#L78" id="L78">
<i class="fa fa-link"></i>
78
</a>
<a class="diff-line-num" data-line-number="79" href="#L79" id="L79">
<i class="fa fa-link"></i>
79
</a>
<a class="diff-line-num" data-line-number="80" href="#L80" id="L80">
<i class="fa fa-link"></i>
80
</a>
<a class="diff-line-num" data-line-number="81" href="#L81" id="L81">
<i class="fa fa-link"></i>
81
</a>
<a class="diff-line-num" data-line-number="82" href="#L82" id="L82">
<i class="fa fa-link"></i>
82
</a>
<a class="diff-line-num" data-line-number="83" href="#L83" id="L83">
<i class="fa fa-link"></i>
83
</a>
<a class="diff-line-num" data-line-number="84" href="#L84" id="L84">
<i class="fa fa-link"></i>
84
</a>
<a class="diff-line-num" data-line-number="85" href="#L85" id="L85">
<i class="fa fa-link"></i>
85
</a>
<a class="diff-line-num" data-line-number="86" href="#L86" id="L86">
<i class="fa fa-link"></i>
86
</a>
<a class="diff-line-num" data-line-number="87" href="#L87" id="L87">
<i class="fa fa-link"></i>
87
</a>
<a class="diff-line-num" data-line-number="88" href="#L88" id="L88">
<i class="fa fa-link"></i>
88
</a>
<a class="diff-line-num" data-line-number="89" href="#L89" id="L89">
<i class="fa fa-link"></i>
89
</a>
<a class="diff-line-num" data-line-number="90" href="#L90" id="L90">
<i class="fa fa-link"></i>
90
</a>
<a class="diff-line-num" data-line-number="91" href="#L91" id="L91">
<i class="fa fa-link"></i>
91
</a>
<a class="diff-line-num" data-line-number="92" href="#L92" id="L92">
<i class="fa fa-link"></i>
92
</a>
<a class="diff-line-num" data-line-number="93" href="#L93" id="L93">
<i class="fa fa-link"></i>
93
</a>
<a class="diff-line-num" data-line-number="94" href="#L94" id="L94">
<i class="fa fa-link"></i>
94
</a>
<a class="diff-line-num" data-line-number="95" href="#L95" id="L95">
<i class="fa fa-link"></i>
95
</a>
<a class="diff-line-num" data-line-number="96" href="#L96" id="L96">
<i class="fa fa-link"></i>
96
</a>
<a class="diff-line-num" data-line-number="97" href="#L97" id="L97">
<i class="fa fa-link"></i>
97
</a>
<a class="diff-line-num" data-line-number="98" href="#L98" id="L98">
<i class="fa fa-link"></i>
98
</a>
<a class="diff-line-num" data-line-number="99" href="#L99" id="L99">
<i class="fa fa-link"></i>
99
</a>
<a class="diff-line-num" data-line-number="100" href="#L100" id="L100">
<i class="fa fa-link"></i>
100
</a>
<a class="diff-line-num" data-line-number="101" href="#L101" id="L101">
<i class="fa fa-link"></i>
101
</a>
<a class="diff-line-num" data-line-number="102" href="#L102" id="L102">
<i class="fa fa-link"></i>
102
</a>
<a class="diff-line-num" data-line-number="103" href="#L103" id="L103">
<i class="fa fa-link"></i>
103
</a>
<a class="diff-line-num" data-line-number="104" href="#L104" id="L104">
<i class="fa fa-link"></i>
104
</a>
<a class="diff-line-num" data-line-number="105" href="#L105" id="L105">
<i class="fa fa-link"></i>
105
</a>
<a class="diff-line-num" data-line-number="106" href="#L106" id="L106">
<i class="fa fa-link"></i>
106
</a>
<a class="diff-line-num" data-line-number="107" href="#L107" id="L107">
<i class="fa fa-link"></i>
107
</a>
<a class="diff-line-num" data-line-number="108" href="#L108" id="L108">
<i class="fa fa-link"></i>
108
</a>
<a class="diff-line-num" data-line-number="109" href="#L109" id="L109">
<i class="fa fa-link"></i>
109
</a>
<a class="diff-line-num" data-line-number="110" href="#L110" id="L110">
<i class="fa fa-link"></i>
110
</a>
<a class="diff-line-num" data-line-number="111" href="#L111" id="L111">
<i class="fa fa-link"></i>
111
</a>
<a class="diff-line-num" data-line-number="112" href="#L112" id="L112">
<i class="fa fa-link"></i>
112
</a>
<a class="diff-line-num" data-line-number="113" href="#L113" id="L113">
<i class="fa fa-link"></i>
113
</a>
<a class="diff-line-num" data-line-number="114" href="#L114" id="L114">
<i class="fa fa-link"></i>
114
</a>
<a class="diff-line-num" data-line-number="115" href="#L115" id="L115">
<i class="fa fa-link"></i>
115
</a>
<a class="diff-line-num" data-line-number="116" href="#L116" id="L116">
<i class="fa fa-link"></i>
116
</a>
<a class="diff-line-num" data-line-number="117" href="#L117" id="L117">
<i class="fa fa-link"></i>
117
</a>
<a class="diff-line-num" data-line-number="118" href="#L118" id="L118">
<i class="fa fa-link"></i>
118
</a>
<a class="diff-line-num" data-line-number="119" href="#L119" id="L119">
<i class="fa fa-link"></i>
119
</a>
<a class="diff-line-num" data-line-number="120" href="#L120" id="L120">
<i class="fa fa-link"></i>
120
</a>
<a class="diff-line-num" data-line-number="121" href="#L121" id="L121">
<i class="fa fa-link"></i>
121
</a>
</div>
<div class="blob-content" data-blob-id="8709c6d9a4c2a713b694990d5b78493b6a67c2fe">
<pre class="code highlight"><code><span id="LC1" class="line"><span class="s">""" Standard fingerprint functions. """</span></span>
<span id="LC2" class="line"><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="n">np</span></span>
<span id="LC3" class="line"></span>
<span id="LC4" class="line"><span class="n">no_asap</span> <span class="o">=</span> <span class="bp">False</span></span>
<span id="LC5" class="line"><span class="k">try</span><span class="p">:</span></span>
<span id="LC6" class="line">    <span class="kn">from</span> <span class="nn">asap3.analysis</span> <span class="kn">import</span> <span class="n">PTM</span></span>
<span id="LC7" class="line"><span class="k">except</span> <span class="nb">ImportError</span><span class="p">:</span></span>
<span id="LC8" class="line">    <span class="n">no_asap</span> <span class="o">=</span> <span class="bp">True</span></span>
<span id="LC9" class="line"></span>
<span id="LC10" class="line"></span>
<span id="LC11" class="line"><span class="k">class</span> <span class="nc">StandardFingerprintGenerator</span><span class="p">(</span><span class="nb">object</span><span class="p">):</span></span>
<span id="LC12" class="line">    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">atom_types</span><span class="o">=</span><span class="bp">None</span><span class="p">):</span></span>
<span id="LC13" class="line">        <span class="s">""" atom_types: list</span></span>
<span id="LC14" class="line"><span class="s">                List of all unique atomic types in the systems under</span></span>
<span id="LC15" class="line"><span class="s">                consideration. Should always be defined if elemental makeup</span></span>
<span id="LC16" class="line"><span class="s">                varies between candidates to preserve a constant ordering.</span></span>
<span id="LC17" class="line"><span class="s">        """</span></span>
<span id="LC18" class="line">        <span class="bp">self</span><span class="o">.</span><span class="n">atom_types</span> <span class="o">=</span> <span class="n">atom_types</span></span>
<span id="LC19" class="line"></span>
<span id="LC20" class="line">    <span class="k">def</span> <span class="nf">mass_fpv</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">atoms</span><span class="p">):</span></span>
<span id="LC21" class="line">        <span class="s">""" Function that takes a list of atoms objects and returns the mass.</span></span>
<span id="LC22" class="line"><span class="s">        """</span></span>
<span id="LC23" class="line">        <span class="c"># Return the summed mass of the atoms object.</span></span>
<span id="LC24" class="line">        <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="nb">sum</span><span class="p">(</span><span class="n">atoms</span><span class="o">.</span><span class="n">get_masses</span><span class="p">())])</span></span>
<span id="LC25" class="line"></span>
<span id="LC26" class="line">    <span class="k">def</span> <span class="nf">composition_fpv</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">atoms</span><span class="p">):</span></span>
<span id="LC27" class="line">        <span class="s">""" Basic function to take atoms object and return the composition. """</span></span>
<span id="LC28" class="line">        <span class="c"># Generate a list of atom types if not supplied.</span></span>
<span id="LC29" class="line">        <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">atom_types</span> <span class="ow">is</span> <span class="bp">None</span><span class="p">:</span></span>
<span id="LC30" class="line">            <span class="bp">self</span><span class="o">.</span><span class="n">atom_types</span> <span class="o">=</span> <span class="nb">frozenset</span><span class="p">(</span><span class="n">atoms</span><span class="o">.</span><span class="n">get_chemical_symbols</span><span class="p">())</span></span>
<span id="LC31" class="line"></span>
<span id="LC32" class="line">        <span class="c"># Add count of each atom type to the fingerprint vector.</span></span>
<span id="LC33" class="line">        <span class="n">fp</span> <span class="o">=</span> <span class="p">[]</span></span>
<span id="LC34" class="line">        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">atom_types</span><span class="p">:</span></span>
<span id="LC35" class="line">            <span class="n">count</span> <span class="o">=</span> <span class="mf">0.</span></span>
<span id="LC36" class="line">            <span class="k">for</span> <span class="n">j</span> <span class="ow">in</span> <span class="n">atoms</span><span class="o">.</span><span class="n">get_chemical_symbols</span><span class="p">():</span></span>
<span id="LC37" class="line">                <span class="k">if</span> <span class="n">i</span> <span class="o">==</span> <span class="n">j</span><span class="p">:</span></span>
<span id="LC38" class="line">                    <span class="n">count</span> <span class="o">+=</span> <span class="mf">1.</span></span>
<span id="LC39" class="line">            <span class="n">fp</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">count</span><span class="p">)</span></span>
<span id="LC40" class="line">        <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">fp</span><span class="p">)</span></span>
<span id="LC41" class="line"></span>
<span id="LC42" class="line">    <span class="k">def</span> <span class="nf">get_coulomb</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">atoms</span><span class="p">):</span></span>
<span id="LC43" class="line">        <span class="s">""" Function to generate the coulomb matrix.</span></span>
<span id="LC44" class="line"></span>
<span id="LC45" class="line"><span class="s">        Returns a numpy array.</span></span>
<span id="LC46" class="line"><span class="s">        """</span></span>
<span id="LC47" class="line">        <span class="c"># Get distances</span></span>
<span id="LC48" class="line">        <span class="n">dm</span> <span class="o">=</span> <span class="n">atoms</span><span class="o">.</span><span class="n">get_all_distances</span><span class="p">()</span></span>
<span id="LC49" class="line">        <span class="n">np</span><span class="o">.</span><span class="n">fill_diagonal</span><span class="p">(</span><span class="n">dm</span><span class="p">,</span> <span class="mf">1.</span><span class="p">)</span></span>
<span id="LC50" class="line"></span>
<span id="LC51" class="line">        <span class="c"># Make coulomb matrix</span></span>
<span id="LC52" class="line">        <span class="n">coulomb</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">outer</span><span class="p">(</span><span class="n">atoms</span><span class="o">.</span><span class="n">numbers</span><span class="p">,</span> <span class="n">atoms</span><span class="o">.</span><span class="n">numbers</span><span class="p">)</span> <span class="o">/</span> <span class="n">dm</span></span>
<span id="LC53" class="line"></span>
<span id="LC54" class="line">        <span class="c"># Set diagonal elements</span></span>
<span id="LC55" class="line">        <span class="n">r</span> <span class="o">=</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">atoms</span><span class="p">))</span></span>
<span id="LC56" class="line">        <span class="n">coulomb</span><span class="p">[</span><span class="n">r</span><span class="p">,</span> <span class="n">r</span><span class="p">]</span> <span class="o">=</span> <span class="mf">0.5</span> <span class="o">*</span> <span class="n">atoms</span><span class="o">.</span><span class="n">numbers</span> <span class="o">**</span> <span class="mf">2.4</span></span>
<span id="LC57" class="line"></span>
<span id="LC58" class="line">        <span class="k">return</span> <span class="n">coulomb</span></span>
<span id="LC59" class="line"></span>
<span id="LC60" class="line">    <span class="k">def</span> <span class="nf">eigenspectrum_fpv</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">atoms</span><span class="p">):</span></span>
<span id="LC61" class="line">        <span class="s">""" Function that takes a list of atoms objects and returns a list of</span></span>
<span id="LC62" class="line"><span class="s">            fingerprint vectors in the form of the sorted eigenspectrum of the</span></span>
<span id="LC63" class="line"><span class="s">            Coulomb matrix as defined in J. Chem. Theory Comput. 2013, 9,</span></span>
<span id="LC64" class="line"><span class="s">            3404-3419.</span></span>
<span id="LC65" class="line"><span class="s">        """</span></span>
<span id="LC66" class="line">        <span class="c"># Get the Coulomb matrix.</span></span>
<span id="LC67" class="line">        <span class="n">coulomb</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">get_coulomb</span><span class="p">(</span><span class="n">atoms</span><span class="p">)</span></span>
<span id="LC68" class="line">        <span class="c"># Get eigenvalues and vectors</span></span>
<span id="LC69" class="line">        <span class="n">w</span><span class="p">,</span> <span class="n">v</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">eig</span><span class="p">((</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">coulomb</span><span class="p">)))</span></span>
<span id="LC70" class="line">        <span class="c"># Return sort eigenvalues from largest to smallest</span></span>
<span id="LC71" class="line">        <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">sort</span><span class="p">(</span><span class="n">w</span><span class="p">)[::</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span></span>
<span id="LC72" class="line"></span>
<span id="LC73" class="line">    <span class="k">def</span> <span class="nf">distance_fpv</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">atoms</span><span class="p">):</span></span>
<span id="LC74" class="line">        <span class="s">""" Function to calculate the averaged distance between e.g. A-A atomic</span></span>
<span id="LC75" class="line"><span class="s">            pairs. The distance measure can be useful to describe how close</span></span>
<span id="LC76" class="line"><span class="s">            atoms preferentially sit in the system.</span></span>
<span id="LC77" class="line"><span class="s">        """</span></span>
<span id="LC78" class="line">        <span class="n">fp</span> <span class="o">=</span> <span class="p">[]</span></span>
<span id="LC79" class="line">        <span class="n">an</span> <span class="o">=</span> <span class="n">atoms</span><span class="o">.</span><span class="n">get_atomic_numbers</span><span class="p">()</span></span>
<span id="LC80" class="line">        <span class="n">pos</span> <span class="o">=</span> <span class="n">atoms</span><span class="o">.</span><span class="n">get_positions</span><span class="p">()</span></span>
<span id="LC81" class="line">        <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">atom_types</span> <span class="ow">is</span> <span class="bp">None</span><span class="p">:</span></span>
<span id="LC82" class="line">            <span class="c"># Get unique atom types.</span></span>
<span id="LC83" class="line">            <span class="bp">self</span><span class="o">.</span><span class="n">atom_types</span> <span class="o">=</span> <span class="nb">frozenset</span><span class="p">(</span><span class="n">an</span><span class="p">)</span></span>
<span id="LC84" class="line">        <span class="k">for</span> <span class="n">at</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">atom_types</span><span class="p">:</span></span>
<span id="LC85" class="line">            <span class="n">ad</span> <span class="o">=</span> <span class="mf">0.</span></span>
<span id="LC86" class="line">            <span class="n">co</span> <span class="o">=</span> <span class="mi">0</span></span>
<span id="LC87" class="line">            <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">j</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">an</span><span class="p">,</span> <span class="n">pos</span><span class="p">):</span></span>
<span id="LC88" class="line">                <span class="k">if</span> <span class="n">i</span> <span class="o">==</span> <span class="n">at</span><span class="p">:</span></span>
<span id="LC89" class="line">                    <span class="k">for</span> <span class="n">k</span><span class="p">,</span> <span class="n">l</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">an</span><span class="p">,</span> <span class="n">pos</span><span class="p">):</span></span>
<span id="LC90" class="line">                        <span class="k">if</span> <span class="n">k</span> <span class="o">==</span> <span class="n">at</span> <span class="ow">and</span> <span class="nb">all</span><span class="p">(</span><span class="n">j</span> <span class="o">!=</span> <span class="n">l</span><span class="p">):</span></span>
<span id="LC91" class="line">                            <span class="n">co</span> <span class="o">+=</span> <span class="mi">1</span></span>
<span id="LC92" class="line">                            <span class="n">ad</span> <span class="o">+=</span> <span class="n">np</span><span class="o">.</span><span class="n">linalg</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span><span class="n">j</span><span class="o">-</span><span class="n">l</span><span class="p">)</span></span>
<span id="LC93" class="line">            <span class="k">if</span> <span class="n">co</span> <span class="o">!=</span> <span class="mi">0</span><span class="p">:</span></span>
<span id="LC94" class="line">                <span class="n">fp</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">ad</span> <span class="o">/</span> <span class="n">co</span><span class="p">)</span></span>
<span id="LC95" class="line">            <span class="k">else</span><span class="p">:</span></span>
<span id="LC96" class="line">                <span class="n">fp</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="mf">0.</span><span class="p">)</span></span>
<span id="LC97" class="line">        <span class="k">return</span> <span class="n">fp</span></span>
<span id="LC98" class="line"></span>
<span id="LC99" class="line">    <span class="k">def</span> <span class="nf">ptm_structure_fpv</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">atoms</span><span class="p">):</span></span>
<span id="LC100" class="line">        <span class="s">""" Function that uses the Polyhedral Template Matching routine in ASAP</span></span>
<span id="LC101" class="line"><span class="s">            to assess the structure and return a list with the crystal</span></span>
<span id="LC102" class="line"><span class="s">            structure environment of each atom. Greater detail can be found at</span></span>
<span id="LC103" class="line"><span class="s">            the following:</span></span>
<span id="LC104" class="line"><span class="s">            https://wiki.fysik.dtu.dk/asap/Local</span><span class="si">%20</span><span class="s">crystalline</span><span class="si">%20</span><span class="s">order</span></span>
<span id="LC105" class="line"><span class="s">        """</span></span>
<span id="LC106" class="line">        <span class="n">msg</span> <span class="o">=</span> <span class="s">"ASAP must be installed to use this function:"</span></span>
<span id="LC107" class="line">        <span class="n">msg</span> <span class="o">+=</span> <span class="s">" https://wiki.fysik.dtu.dk/asap"</span></span>
<span id="LC108" class="line">        <span class="k">assert</span> <span class="ow">not</span> <span class="n">no_asap</span><span class="p">,</span> <span class="n">msg</span></span>
<span id="LC109" class="line">        <span class="n">ptmdata</span> <span class="o">=</span> <span class="n">PTM</span><span class="p">(</span><span class="n">atoms</span><span class="p">)</span></span>
<span id="LC110" class="line">        <span class="k">return</span> <span class="n">ptmdata</span><span class="p">[</span><span class="s">'structure'</span><span class="p">]</span></span>
<span id="LC111" class="line"></span>
<span id="LC112" class="line">    <span class="k">def</span> <span class="nf">ptm_alloy_fpv</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">atoms</span><span class="p">):</span></span>
<span id="LC113" class="line">        <span class="s">""" Function that uses the Polyhedral Template Matching routine in ASAP</span></span>
<span id="LC114" class="line"><span class="s">            to assess the structure and return a list with the alloy structure</span></span>
<span id="LC115" class="line"><span class="s">            environment of each atom.</span></span>
<span id="LC116" class="line"><span class="s">        """</span></span>
<span id="LC117" class="line">        <span class="n">msg</span> <span class="o">=</span> <span class="s">"ASAP must be installed to use this function:"</span></span>
<span id="LC118" class="line">        <span class="n">msg</span> <span class="o">+=</span> <span class="s">" https://wiki.fysik.dtu.dk/asap"</span></span>
<span id="LC119" class="line">        <span class="k">assert</span> <span class="ow">not</span> <span class="n">no_asap</span><span class="p">,</span> <span class="n">msg</span></span>
<span id="LC120" class="line">        <span class="n">ptmdata</span> <span class="o">=</span> <span class="n">PTM</span><span class="p">(</span><span class="n">atoms</span><span class="p">)</span></span>
<span id="LC121" class="line">        <span class="k">return</span> <span class="n">ptmdata</span><span class="p">[</span><span class="s">'alloytype'</span><span class="p">]</span></span></code></pre>
</div>
</div>


</article>
</div>

</div>
</div>

</div>
</div>
</div>
</div>



</body>
</html>

